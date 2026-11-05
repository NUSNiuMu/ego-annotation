"""Qwen adapter copied from the local baseline, with no experiment runner dependency."""
import hashlib
import json
import os
import time
import uuid
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

from ego_annotation.inference.attention import RUNTIME_ID, turing_sdpa
from ego_annotation.inference.schema import parse_response, prompt
from ego_annotation.config import ASSISTANT_PREFIX, SYSTEM_PROMPT
from ego_annotation.inference.acceleration import DispatchProbe, versions as acceleration_versions


def build_messages(indices, samples):
    content = []
    for i in indices:
        content.extend([
            {'type': 'text', 'text': f"frame={i}; relative_time={samples[i]['relative_s']:.9f}s"},
            {'type': 'image'},
        ])
    content.append({'type': 'text', 'text': prompt(indices)})
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]


def verify_model(model_dir):
    """Record differences from copied baseline; the model loader checks compatibility."""
    model_dir = Path(model_dir)
    identity = json.loads(Path(__file__).with_name('model_identity.json').read_text())
    differences = []
    observed = {}
    for name, expected in identity['metadata'].items():
        p = model_dir / name
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        observed[name] = actual
        if actual != expected['sha256']:
            differences.append({'file': name, 'expected_sha256': expected['sha256'], 'actual_sha256': actual})
    for name, expected in identity['weights'].items():
        p = model_dir / name
        actual = p.stat().st_size if p.is_file() else None
        if actual != expected['bytes']:
            differences.append({'file': name, 'expected_bytes': expected['bytes'], 'actual_bytes': actual})
    return {'baseline_revision': identity['revision'], 'metadata_sha256': observed,
            'differences': differences, 'policy': 'record_only'}


class QwenWorker:
    def __init__(self, model_dir, gpu, config):
        self.config = config
        self.identity = verify_model(model_dir)
        expected = {'torch': '2.8.0', 'transformers': '5.19.0'}
        actual = {name: version(name) for name in expected}
        version_differences = {name: {'baseline': release, 'actual': actual[name]}
                               for name, release in expected.items() if actual[name].split('+')[0] != release}
        from ..gpu_devices import configure
        gpu_uuid = configure(gpu)
        os.environ.update(HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                          TOKENIZERS_PARALLELISM='false')
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.torch = torch
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(0)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, device_map={'': 'cuda:0'}, attn_implementation=config.attention,
        ).eval()
        torch.cuda.synchronize()
        if str(torch.cuda.get_device_properties(0).uuid) != gpu_uuid.removeprefix('GPU-'):
            raise RuntimeError('Loaded CUDA device differs from assigned GPU UUID')
        self.runtime = {
            'model': config.model_name, 'baseline_revision': config.model_revision,
            'model_directory': str(Path(model_dir).resolve()), 'versions': actual,
            'gpu': gpu, 'gpu_uuid': gpu_uuid, 'gpu_name': torch.cuda.get_device_name(),
            'attention_runtime': RUNTIME_ID, 'load_s': time.perf_counter() - started,
            'weight_verification': 'metadata SHA256 and shard sizes; no weight-content hash claim',
            'model_identity': self.identity, 'version_differences': version_differences,
            'acceleration_packages': acceleration_versions(),
        }
        self._probe_pending = True
        self._warmup()

    @property
    def pixel_budget(self):
        return {'shortest_edge': self.config.processor_min_pixels,
                'longest_edge': self.config.processor_max_pixels}

    def _warmup(self):
        from PIL import Image
        torch = self.torch
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Say OK.'}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        started = time.perf_counter()
        inputs = self.processor(text=[text], images=[Image.new('RGB', (64, 64))],
                                return_tensors='pt', size=self.pixel_budget).to('cuda')
        with torch.inference_mode(), turing_sdpa(RUNTIME_ID):
            self.model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        self.runtime['warmup_s'] = time.perf_counter() - started

    def infer(self, window, samples, images):
        return self.infer_batch([{'window': window, 'samples': samples, 'images': images}])[0]

    def infer_batch(self, requests):
        """One generate call owns all model/global attention state; never use threads."""
        if not 1 <= len(requests) <= 2:
            raise ValueError('Qwen accepts one or two independent windows')
        torch = self.torch
        texts, messages, images = [], [], []
        for request in requests:
            message = build_messages(request['window']['indices'], request['samples'])
            messages.append(message)
            texts.append(self.processor.apply_chat_template(message, tokenize=False,
                add_generation_prompt=True, enable_thinking=self.config.enable_thinking) + ASSISTANT_PREFIX)
            images.extend(request['images'])
        started = time.perf_counter()
        inputs = self.processor(text=texts, images=images, padding=True,
                                return_tensors='pt', size=self.pixel_budget)
        processor_s = time.perf_counter() - started
        grids = inputs['image_grid_thw'].tolist()
        input_width = inputs['input_ids'].shape[-1]
        input_tokens = inputs['attention_mask'].sum(dim=1).tolist()
        input_hashes = [hashlib.sha256(ids[mask.bool()].numpy().tobytes()).hexdigest()
                       for ids, mask in zip(inputs['input_ids'], inputs['attention_mask'])]
        tick = time.perf_counter()
        inputs = inputs.to('cuda')
        torch.cuda.synchronize()
        h2d_s = time.perf_counter() - tick
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        probe = DispatchProbe() if self._probe_pending else None
        with torch.inference_mode(), turing_sdpa(RUNTIME_ID) as attention_stats, (probe or nullcontext()):
            generated = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample, use_cache=self.config.use_cache,
            )
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - tick
        if probe is not None:
            self.runtime['acceleration_dispatch'] = probe.report()
            self._probe_pending = False
        output_end = time.perf_counter()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        results, offset, count = [], 0, len(requests)
        batch_id = uuid.uuid4().hex
        for i, (request, ids) in enumerate(zip(requests, generated[:, input_width:])):
            tokens = ids.detach().cpu().tolist()
            stop = next((j+1 for j, token in enumerate(tokens) if token in eos), len(tokens))
            tokens = tokens[:stop]
            continuation = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            raw = ASSISTANT_PREFIX + continuation
            special = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
            indices = request['window']['indices']
            image_grids = grids[offset:offset+len(indices)]
            offset += len(indices)
            truncated = len(tokens) >= self.config.max_new_tokens and (not tokens or tokens[-1] not in eos)
            parsed, error, valid, strict, removed = parse_response(raw, indices)
            results.append({
                'status': 'TRUNCATED' if truncated else ('PASS' if error is None else 'INVALID_JSON_OR_SCHEMA'),
                'source_window': request['window'], 'input_frames': indices,
                'input_ids_sha256': input_hashes[i], 'image_grid_thw': image_grids,
                'visual_tokens': sum(t*h*w//4 for t,h,w in image_grids),
                'input_tokens': int(input_tokens[i]), 'output_tokens': len(tokens),
                'cpu_processor_s': processor_s/count, 'h2d_s': h2d_s/count,
                'processor_to_output_wall_s': (output_end-started)/count,
                'generate_prefill_decode_s': generation_s/count,
                'batch': {'id': batch_id, 'size': count, 'wall_s': output_end-started,
                          'time_accounting': 'equal_share_per_window'},
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'prompt_messages': messages[i], 'processor_text': texts[i],
                'assistant_prefix': ASSISTANT_PREFIX, 'raw_generated_continuation': continuation,
                'raw_response': raw, 'output_token_ids': tokens,
                'thinking_evidence': {'enable_thinking': self.config.enable_thinking,
                    'generated_think_marker_present': '<think>' in special or '</think>' in special},
                'parsed': parsed, 'schema_error': error, 'json_syntax_valid': valid,
                'strict_json': strict, 'parsed_after_wrapper_removal': removed,
                'truncated': truncated, 'attention_stats': attention_stats,
                'dispatch_profiled': probe is not None})
        return results
