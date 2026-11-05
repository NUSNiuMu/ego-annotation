"""Real local MCAP/decode and durable review, with explicitly fake model calls."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ego_annotation import cli, pipeline, review, review_runner, sampling
from ego_annotation.config import PipelineConfig
from test_pipeline import FakeWorker, make_source


class CandidateModel(FakeWorker):
    runtime = {'test_double': True, 'gpu_uuid': 'GPU-fixture'}

    def infer(self, window, samples, images):
        value = super().infer(window, samples, images)
        segments = []
        if window['indices'][0] == 0:
            segments = [{'start_frame': 0, 'end_frame': 5, 'action': 'smooth_garment',
                         'description': 'Test-only candidate.', 'evidence_frames': [0, 3, 5], 'continues': 'none'}]
        value['parsed'] = {'segments': segments, 'uncertainties': []}
        value['raw_response'] = json.dumps(value['parsed'])
        return value

    def infer_batch(self, requests):
        return [self.infer(r['window'], r['samples'], r['images']) for r in requests]


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.method = review.ROOT / 'skills'
        self.config = {'model': 'test-model', 'reasoning_effort': 'high', 'codex_binary': 'test-only',
                       'call_timeout_seconds': 10, 'transport_retry_delays_seconds': []}
        self.calls = []
        self.image_issue = False

    def job(self, count=8):
        raw = make_source(self.root / 'raw', count=count)
        with contextlib.redirect_stdout(io.StringIO()):
            row = pipeline.run_batch([[raw]], self.root / 'work', PipelineConfig(), lambda: CandidateModel())[0]
        self.assertEqual(row['status'], 'COMPLETE')
        return raw, Path(row['job_dir'])

    def fake_call(self, config, folder, prompt, schema, *, image_paths=()):
        packet = json.loads(prompt.rsplit('\nINPUT DATA\n', 1)[1])
        stage = schema['properties']['review_type']['enum'][0]
        self.calls.append(stage)
        episode = {'id': 'candidate', 'start_frame': 0, 'end_frame_exclusive': 6,
                   'disposition': 'ACCEPT', 'reason': 'Test-only complete task.', 'boundary_notes': '',
                   'evidence_refs': ['w000:s00'], 'unlabeled': [], 'issues': [],
                   'actions': [{'code': 'smooth_garment', 'start_frame': 0, 'end_frame_exclusive': 6,
                                'description': 'Test-only action.', 'source_refs': ['w000:s00'],
                                'evidence_frames': [0, 3, 5], 'edit': 'retain', 'reason': 'Fixture evidence.'}]}
        result = {'packet_id': packet['packet_id'], 'review_type': stage, 'episodes': [episode], 'excluded': []}
        if stage == 'text' and self.image_issue:
            episode['disposition'] = 'REVIEW'
            episode['issues'] = [{'id': 'completion', 'kind': 'COMPLETION', 'question': 'Test-only completion issue.',
                                 'start_frame': 0, 'end_frame_exclusive': 6,
                                 'source_refs': ['w000:s00'], 'requested_frames': [0, 3, 5]}]
        if stage == 'image':
            self.assertEqual(len(image_paths), 3)
            self.assertTrue(all(Path(p).is_file() for p in image_paths))
            episode['source_episode_ids'] = [packet['episodes'][0]['id']]
            result['issue_results'] = [{'issue_id': 'completion', 'resolution': 'RESOLVED',
                                       'observation': 'Test-only image finding.', 'evidence_frames': [0, 3, 5]}]
        folder.mkdir(parents=True)
        review.write_new(folder / 'response.json', result)
        review.write_new(folder / 'call-receipt.json', {'returncode': 0, 'response': review.pin(folder / 'response.json')})
        return result, {'test_double': True}

    def test_full_batch_uses_local_inputs_and_reuses_every_completed_result(self):
        raw = make_source(self.root / 'raw')
        other = make_source(self.root / 'other', count=20)
        model = CandidateModel()
        config = {'qwen': {'model': str(self.root / 'model'), 'gpu': 0},
                  'review': self.config, 'method_dir': str(self.method)}
        with patch.object(cli, 'device', return_value={'uuid': 'GPU-fixture'}), \
                patch.object(cli.subprocess, 'check_output', return_value=''), \
                patch.object(pipeline, 'model_fingerprint', return_value={}), \
                patch('ego_annotation.inference.worker.QwenWorker', return_value=model) as factory, \
                patch.object(review_runner, 'invoke', self.fake_call), contextlib.redirect_stdout(io.StringIO()):
            rows = cli.run([[raw], [other]], self.root / 'work', config, PipelineConfig())
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['review']['accepted_episodes'] == 1 for r in rows))
            pins = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.root / 'work').glob('jobs/*/windows/*.json')}
            before = len(model.calls)
            again = cli.run([[raw], [other]], self.root / 'work', config, PipelineConfig())
            self.assertEqual(factory.call_count, 1)
        self.assertTrue(all(r['review']['status'] == 'COMPLETE' for r in again))
        self.assertEqual(self.calls, ['text', 'text'])
        self.assertEqual(len(model.calls), before)
        for name, expected in pins.items():
            self.assertEqual((Path(name).read_bytes(), Path(name).stat().st_mtime_ns), expected)

    def test_targeted_images_only_for_requested_issue(self):
        _, job = self.job()
        self.image_issue = True
        with patch.object(review_runner, 'invoke', self.fake_call):
            result = review_runner.run_review(job, self.config, self.method)
        self.assertEqual(self.calls, ['text', 'image'])
        self.assertEqual(result['accepted_episodes'], 1)
        final = review.read(Path(result['case']) / 'final/episodes.json')
        self.assertEqual(final['episodes'][0]['evidence_basis'], 'targeted_image_review')

    def test_completed_review_tampering_is_rejected_without_a_model_call(self):
        _, job = self.job()
        with patch.object(review_runner, 'invoke', self.fake_call):
            result = review_runner.run_review(job, self.config, self.method)
        (Path(result['case']) / 'final/episodes.json').write_text('{}')
        with patch.object(review_runner, 'invoke', side_effect=AssertionError('No retry')):
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                review_runner.run_review(job, self.config, self.method)

    def test_partial_recording_failure_does_not_block_other_input(self):
        raw = make_source(self.root / 'raw')
        (self.root / 'broken').mkdir()
        broken = self.root / 'broken/part.mcap'
        broken.write_bytes(b'not MCAP')
        config = {'qwen': {'model': str(self.root / 'model'), 'gpu': 0},
                  'review': self.config, 'method_dir': str(self.method)}
        with patch.object(cli, 'device', return_value={'uuid': 'GPU-fixture'}), \
                patch.object(cli.subprocess, 'check_output', return_value=''), \
                patch.object(pipeline, 'model_fingerprint', return_value={}), \
                patch('ego_annotation.inference.worker.QwenWorker', return_value=CandidateModel()), \
                patch.object(review_runner, 'invoke', self.fake_call), contextlib.redirect_stdout(io.StringIO()):
            rows = cli.run([[broken], [raw]], self.root / 'work', config, PipelineConfig())
        self.assertEqual(sorted(r['status'] for r in rows), ['COMPLETE', 'FAILED'])
        self.assertEqual(next(r for r in rows if r['status'] == 'COMPLETE')['review']['accepted_episodes'], 1)

    def test_config_resolves_paths_and_rejects_unknown_sampling_setting(self):
        example = review.ROOT / 'configs/annotation.example.json'
        config, params = cli.load_config(example)
        self.assertEqual(Path(config['method_dir']), self.method)
        config['qwen']['sampling']['typo'] = True
        path = self.root / 'local.json'
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'Unknown sampling'):
            cli.load_config(path)

    def test_clock_jump_is_rejected_before_window_allocation(self):
        info = {'median_period_ns': 33_333_333}
        meta = [{'stamp_ns': 0, 'log_ns': 0}, {'stamp_ns': 10**18, 'log_ns': 33_333_333}]
        with patch.object(sampling, 'inspect_source', return_value=(info, meta)), \
                patch.object(sampling, 'plan_windows', side_effect=AssertionError('No allocation')):
            with self.assertRaisesRegex(ValueError, 'TIMESTAMP_CLOCK_JUMP'):
                sampling.run([], 'test-only', self.root / 'prepared')

    def test_saved_successful_call_is_recovered_without_calling_again(self):
        _, job = self.job()
        def interrupted(*args, **kwargs):
            self.fake_call(*args, **kwargs)
            raise KeyboardInterrupt('Test-only interruption after the response was saved')
        with patch.object(review_runner, 'invoke', interrupted):
            with self.assertRaises(KeyboardInterrupt):
                review_runner.run_review(job, self.config, self.method)
        with patch.object(review_runner, 'invoke', side_effect=AssertionError('No repeat')):
            result = review_runner.run_review(job, self.config, self.method)
        self.assertEqual(result['accepted_episodes'], 1)
        self.assertEqual(self.calls, ['text'])

    def test_bad_qwen_window_forces_targeted_review_without_inference_replay(self):
        raw = make_source(self.root / 'raw')
        model = FakeWorker(fail_at=0)
        with contextlib.redirect_stdout(io.StringIO()):
            row = pipeline.run_batch([[raw]], self.root / 'work', PipelineConfig(), lambda: model)[0]
        job = Path(row['job_dir'])
        original = (job / 'windows/000000.json').read_bytes()
        def gap_call(config, folder, prompt, schema, *, image_paths=()):
            packet = json.loads(prompt.rsplit('\nINPUT DATA\n', 1)[1])
            stage = schema['properties']['review_type']['enum'][0]
            self.calls.append(stage)
            value = {'packet_id': packet['packet_id'], 'review_type': stage, 'episodes': [],
                     'excluded': [{'source_ref': 'w000:gap', 'reason': 'Test-only non-task context.'}]}
            if stage == 'image':
                self.assertTrue(image_paths)
                value['issue_results'] = [{'issue_id': i['id'], 'resolution': 'RESOLVED',
                    'observation': 'Test-only non-task evidence.', 'evidence_frames': [0, 3, 5]} for i in packet['issues']]
            folder.mkdir(parents=True)
            review.write_new(folder / 'response.json', value)
            review.write_new(folder / 'call-receipt.json', {'returncode': 0, 'response': review.pin(folder / 'response.json')})
            return value, {'test_double': True}
        with patch.object(review_runner, 'invoke', gap_call):
            result = review_runner.run_review(job, self.config, self.method)
        self.assertEqual(self.calls, ['text', 'image'])
        self.assertEqual(result['deferred_episodes'], 0, review.read(Path(result['case']) / 'final/deferred.json'))
        self.assertEqual((job / 'windows/000000.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
