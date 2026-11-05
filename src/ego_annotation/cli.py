"""Local MCAP -> Qwen dual input -> fixed skill review -> saved annotations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import tempfile

from . import pipeline
from .config import PipelineConfig
from .gpu_devices import device
from .qwen_batch import run_batched
from .review_runner import run_review


def load_config(path):
    path = Path(path).resolve()
    config = pipeline.read(path)
    if set(config) != {'qwen', 'review', 'method_dir'}:
        raise ValueError('Config requires qwen, review and method_dir')
    for key in ('model', 'gpu', 'sampling'):
        if key not in config['qwen']:
            raise ValueError('Missing qwen.' + key)
    allowed = {f.name for f in fields(PipelineConfig) if f.init}
    unknown = set(config['qwen']['sampling']) - allowed
    if unknown:
        raise ValueError('Unknown sampling/model settings: ' + str(sorted(unknown)))
    params = PipelineConfig(**config['qwen']['sampling'])
    if (params.window_s <= 0 or params.interval_s <= 0 or params.interval_s > params.window_s or params.longest_edge <= 0
            or not 1 <= params.jpeg_quality <= 100 or params.decode_threads < 1
            or params.cpu_threads < 1 or params.max_new_tokens < 1):
        raise ValueError('Invalid sampling/model settings')
    for owner, key in ((config['qwen'], 'model'), (config, 'method_dir')):
        value = Path(owner[key]).expanduser()
        owner[key] = str(value.resolve() if value.is_absolute() else (path.parent / value).resolve())
    if type(config['qwen']['gpu']) is not int or config['qwen']['gpu'] < 0:
        raise ValueError('qwen.gpu must be a nonnegative local GPU index')
    for key in ('model', 'reasoning_effort', 'codex_binary', 'call_timeout_seconds'):
        if key not in config['review']:
            raise ValueError('Missing review.' + key)
    if config['review']['call_timeout_seconds'] <= 0:
        raise ValueError('Review timeout must be positive')
    return config, params


def run(groups, output, config, params, *, prepare_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = Path(config['qwen']['model'])
    for parent in [model, *[parts[0].parent for parts in groups]]:
        if output == parent or parent in output.parents:
            raise ValueError('Output must be separate from raw inputs and model weights')
    with pipeline.lock(output / '.controller.lock'):
        if prepare_only:
            return pipeline.run_batch(groups, output, params, lambda: None, prepare_only=True)
        method = Path(config['method_dir'])
        from .review import METHOD_FILES
        for name in METHOD_FILES:
            if not (method / name).is_file():
                raise ValueError('Missing review method file: ' + str(method / name))
        code, model_id = pipeline.runtime_fingerprint(), pipeline.model_fingerprint(model)
        pending = iter(enumerate(groups))
        records, reviews = {}, {}
        worker = None
        with ExitStack() as stack, ThreadPoolExecutor(max_workers=2, thread_name_prefix='review') as pool:
            def get_worker():
                nonlocal worker
                if worker is None:
                    uuid = device(config['qwen']['gpu'])['uuid']
                    stack.enter_context(pipeline.lock(Path(tempfile.gettempdir()) / f'ego-annotation-{uuid}.lock'))
                    busy = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
                        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
                    if busy.strip():
                        raise RuntimeError('GPU already has compute processes; none were changed')
                    from .inference.worker import QwenWorker
                    worker = QwenWorker(model, config['qwen']['gpu'], params)
                    if worker.runtime['gpu_uuid'] != uuid:
                        raise RuntimeError('GPU assignment changed during model startup')
                return worker

            def admit():
                item = next(pending, None)
                if item is None:
                    return None
                key, parts = item
                def compute(proxy):
                    try:
                        return pipeline.process_job(parts, output, params, code, model_id, lambda: proxy)
                    except Exception as error:
                        return {'status': 'FAILED', 'source': [str(p) for p in parts], 'error': str(error)}
                return key, compute

            def complete(key, result):
                records[key] = result
                if result['status'] == 'COMPLETE':
                    reviews[key] = pool.submit(run_review, result['job_dir'], config['review'], method)
                pipeline.save(output / 'batch.json', {'recordings': list(records.values())})

            try:
                run_batched(admit, get_worker, complete)
            except Exception as error:
                pipeline.save(output / 'failure.json', {'error': str(error), 'automatic_retry': False})
                raise
            for key, future in reviews.items():
                try:
                    records[key]['review'] = future.result()
                except Exception as error:
                    records[key]['review'] = {'status': 'FAILED', 'error': str(error)}
                pipeline.save(output / 'batch.json', {'recordings': list(records.values())})
        return list(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--source', type=Path, action='append', default=[], help='Local MCAP file or recording directory; repeatable')
    parser.add_argument('--batch-root', type=Path, action='append', default=[], help='Group MCAP parts by directory recursively')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Sample and validate inputs without calling Qwen or GPT')
    args = parser.parse_args()
    try:
        config, params = load_config(args.config)
        results = run(pipeline.discover(args.source, args.batch_root), args.output, config, params,
                      prepare_only=args.prepare_only)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps({'recordings': results}, ensure_ascii=False))
    return int(any(r['status'] == 'FAILED' or r.get('review', {}).get('status') == 'FAILED' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
