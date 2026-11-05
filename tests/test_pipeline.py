"""Exercise real sampling, batch recovery and immutable completed checkpoints on tiny MCAPs."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import av
from mcap.writer import Writer
from PIL import Image

from ego_annotation import pipeline
from ego_annotation.config import PipelineConfig
from ego_annotation.inference.worker import build_messages, verify_model
from ego_annotation.inference.schema import prompt


def cdr_image(stamp, payload):
    data = bytearray(b'\x00\x01\x00\x00' + struct.pack('<iI', stamp // 10**9, stamp % 10**9))
    for item in (b'camera_rgb\0', b'h264\0'):
        data.extend(b'\0' * ((-(len(data) - 4)) % 4))
        data.extend(struct.pack('<I', len(item)) + item)
    data.extend(b'\0' * ((-(len(data) - 4)) % 4))
    data.extend(struct.pack('<I', len(payload)) + payload)
    return bytes(data)


def make_source(folder, count=8):
    folder.mkdir()
    encoder = av.CodecContext.create('libx264', 'w')
    encoder.width = 64
    encoder.height = 96
    encoder.pix_fmt = 'yuv420p'
    encoder.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
    packets = []
    for i in range(count):
        frame = av.VideoFrame.from_image(Image.new('RGB', (64, 96), (i * 20, 50, 80)))
        packets.extend(encoder.encode(frame))
    packets.extend(encoder.encode(None))
    path = folder / 'part_001.mcap'
    with path.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('sensor_msgs/msg/CompressedImage', 'ros2msg', b'')
        channel = writer.register_channel('/egohalo/record/cam0/h264', 'cdr', schema)
        for i, packet in enumerate(packets):
            stamp = 6_000_000_000_000 + i * 500_000_000
            writer.add_message(channel, stamp + 10_000, cdr_image(stamp, bytes(packet)), stamp + 20_000, i)
        writer.finish()
    return path


class FakeWorker:
    """Test double only: never represents measured Qwen output."""
    runtime = {'test_double': True}

    def __init__(self, fail_at=None, interrupt_at=None):
        self.calls = []
        self.fail_at = fail_at
        self.interrupt_at = interrupt_at

    def infer(self, window, samples, images):
        index = len(self.calls)
        self.calls.append(window['indices'])
        if index == self.interrupt_at:
            raise KeyboardInterrupt('test interruption')
        if index == self.fail_at:
            raw = '{'
            parsed = None
            status = 'INVALID_JSON_OR_SCHEMA'
        else:
            raw = '{"segments": [], "uncertainties": []}'
            parsed = json.loads(raw)
            status = 'PASS'
        if len(images) != len(window['indices']) or any(image.mode != 'RGB' for image in images):
            raise AssertionError('Actual decoded RGB images were not delivered')
        return {'status': status, 'source_window': window, 'raw_response': raw,
                'parsed': parsed, 'truncated': False, 'output_tokens': 10}

    def infer_batch(self, requests):
        return [self.infer(r['window'], r['samples'], r['images']) for r in requests]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = make_source(self.root / 'recording')
        self.work = self.root / 'work'
        self.config = PipelineConfig()

    def tearDown(self):
        self.temp.cleanup()

    def run_batch(self, worker, **kwargs):
        groups = pipeline.discover([self.source.parent], [])
        return pipeline.run_batch(groups, self.work, self.config, lambda: worker, **kwargs)

    def test_prepare_then_inference_then_deduplicate(self):
        prep = self.run_batch(None, prepare_only=True)[0]
        self.assertEqual(prep['status'], 'PREPARED')
        worker = FakeWorker()
        done = self.run_batch(worker)[0]
        self.assertEqual(done['status'], 'COMPLETE')
        self.assertTrue(done['preparation_reused'])
        self.assertEqual(len(worker.calls), 2)
        windows = sorted((Path(done['job_dir']) / 'windows').glob('*.json'))
        prior = [(p.read_bytes(), p.stat().st_mtime_ns) for p in windows]
        # No worker creation is allowed when all window checkpoints are valid.
        def forbidden():
            raise AssertionError('Completed data must not invoke the model')
        again = pipeline.run_batch(pipeline.discover([self.source.parent], []), self.work,
                                   self.config, forbidden)[0]
        self.assertEqual(again['reused_windows'], 2)
        self.assertEqual(prior, [(p.read_bytes(), p.stat().st_mtime_ns) for p in windows])

    def test_interrupt_resumes_only_missing_window(self):
        worker = FakeWorker(interrupt_at=1)
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(worker)
        saved = list((self.work / 'jobs').glob('*/windows/*.json'))
        self.assertEqual(len(saved), 1)
        original = saved[0].read_bytes()
        resumed = FakeWorker()
        result = self.run_batch(resumed)[0]
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(len(resumed.calls), 1)
        self.assertEqual(original, saved[0].read_bytes())

    def test_bad_window_continues_and_is_not_replayed(self):
        done = self.run_batch(FakeWorker(fail_at=0))[0]
        self.assertEqual(done['status'], 'COMPLETE')
        self.assertEqual(done['review_windows'], [0])
        saved = sorted((Path(done['job_dir'])/'windows').glob('*.json'))
        before = [p.read_bytes() for p in saved]
        idle = FakeWorker()
        self.assertEqual(self.run_batch(idle)[0]['reused_windows'], 2)
        self.assertEqual(idle.calls, [])
        self.assertEqual(before, [p.read_bytes() for p in saved])
        self.assertEqual(pipeline.read(saved[0])['raw_response'], '{')

    def test_untrusted_checkpoint_stops_without_repeating_inference(self):
        done = self.run_batch(FakeWorker())[0]
        path = Path(done['job_dir'])/'windows/000000.json'
        value = pipeline.read(path); value['binding']['source_id'] = 'wrong'
        pipeline.save(path,value)
        worker = FakeWorker()
        self.assertEqual(self.run_batch(worker)[0]['status'],'FAILED')
        self.assertEqual(worker.calls,[])

    def test_reuse_original_windows_under_new_code_identity(self):
        from unittest.mock import patch
        worker = FakeWorker(fail_at=0,interrupt_at=1)
        with self.assertRaises(KeyboardInterrupt): self.run_batch(worker)
        old = next((self.work/'jobs').iterdir())
        original = {p.name:p.read_bytes() for p in [old/'job.json',*old.glob('windows/*.json')]}
        with patch.object(pipeline,'runtime_fingerprint',return_value={**pipeline.runtime_fingerprint(),'pipeline.py':'new-code'}):
            resumed = FakeWorker()
            done = self.run_batch(resumed,reuse_job=old)[0]
            self.assertEqual(done['status'],'COMPLETE')
            self.assertNotEqual(done['job_id'],old.name)
            self.assertEqual(len(resumed.calls),1)
            self.assertEqual(done['review_windows'],[0])
            current = Path(done['job_dir'])
            old_result = pipeline.read(old/'windows/000000.json')
            new_result = pipeline.checked_result(current/'windows/000000.json',pipeline.read(current/'job.json')['binding'],old_result['source_window'])
            self.assertEqual(new_result['raw_response'],old_result['raw_response'])
            idle = FakeWorker()
            self.assertEqual(self.run_batch(idle,reuse_job=old)[0]['reused_windows'],2)
            self.assertEqual(idle.calls,[])
            (current/'reused/000000.json').write_text('{}')
            with self.assertRaises(ValueError):
                pipeline.checked_result(current/'windows/000000.json',new_result['binding'],old_result['source_window'])
        for p in [old/'job.json',*old.glob('windows/*.json')]:self.assertEqual(p.read_bytes(),original[p.name])

    def test_corrupt_jpeg_is_regenerated_without_repeating_valid_inference(self):
        self.run_batch(FakeWorker())
        image = next((self.work / 'sources').glob('*/prepare-*/frames/*.jpg'))
        image.write_bytes(b'corrupt')
        worker = FakeWorker()
        result = self.run_batch(worker)[0]
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertFalse(result['preparation_reused'])
        self.assertEqual(worker.calls, [])
        self.assertNotEqual(image.read_bytes(), b'corrupt')

    def test_batch_discovery_and_content_deduplication(self):
        import shutil
        copy = self.root / 'copy'
        copy.mkdir()
        shutil.copyfile(self.source, copy / 'different_name.mcap')
        groups = pipeline.discover([], [self.root])
        self.assertEqual(len(groups), 2)
        worker = FakeWorker()
        rows = pipeline.run_batch(groups, self.work, self.config, lambda: worker)
        self.assertEqual(rows[0]['job_id'], rows[1]['job_id'])
        self.assertTrue(rows[1]['duplicate_in_batch'])
        self.assertEqual(len(worker.calls), 2)

    def test_jpeg_input_is_fixed_and_resumes(self):
        with self.assertRaises(TypeError):
            PipelineConfig(transport='rgb-memory')
        self.assertEqual(self.config.snapshot()['transport'], 'jpeg-disk')
        worker = FakeWorker()
        self.assertEqual(self.run_batch(worker)[0]['status'], 'COMPLETE')
        self.assertEqual(len(worker.calls), 2)
        self.assertTrue(list((self.work / 'sources').rglob('*.jpg')))
        self.assertEqual(self.run_batch(FakeWorker())[0]['reused_windows'], 2)

    def test_baseline_prompt_and_nonblocking_model_record(self):
        indices = [0, 15]
        messages = build_messages(indices, {0: {'relative_s': 0}, 15: {'relative_s': .5}})
        self.assertEqual(messages[-1]['content'][-1]['text'], prompt(indices))
        self.assertIn('clothing-handling', messages[-1]['content'][-1]['text'])
        empty_model = self.root / 'model'
        empty_model.mkdir()
        observed = verify_model(empty_model)
        self.assertEqual(observed['policy'], 'record_only')
        self.assertTrue(observed['differences'])

    def test_one_failed_recording_does_not_stop_batch(self):
        other = make_source(self.root / 'other', count=10)
        worker = FakeWorker(fail_at=0)
        rows = pipeline.run_batch([[self.source], [other]], self.work, self.config, lambda: worker)
        self.assertEqual([r['status'] for r in rows], ['COMPLETE', 'COMPLETE'])
        self.assertEqual(rows[0]['review_windows'],[0])
        self.assertEqual(pipeline.read(self.work / 'batch.json')['failed'], 0)

    def test_generated_runtime_is_local_and_not_symlinked(self):
        import ast
        for name in pipeline.RUNTIME_FILES:
            path = pipeline.ROOT / name
            self.assertFalse(path.is_symlink())
            if path.suffix != '.py':
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('robot_dataset_annotator', 'workflow_v1')))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn('/opt/rda', node.value)
                    self.assertNotIn('/home/niumu/data', node.value)
                    self.assertNotIn('inference_results/', node.value)


if __name__ == '__main__':
    unittest.main()
