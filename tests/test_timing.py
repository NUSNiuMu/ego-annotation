"""Timing must preserve real failures, parentage and credential boundaries."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ego_annotation.timing import span


class TimingTests(unittest.TestCase):
    def test_nested_failure_is_recorded_and_original_exception_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'timing.jsonl'
            error = ValueError('original failure')
            with self.assertRaises(ValueError) as caught:
                with span(path, 'parent'):
                    with span(path, 'child') as fields:
                        fields['bytes'] = 123
                        raise error
            self.assertIs(caught.exception, error)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            parent, child, child_end, parent_end = rows
            self.assertEqual(child['parent_span_id'], parent['span_id'])
            self.assertEqual(child_end['status'], 'FAILED')
            self.assertEqual(child_end['bytes'], 123)
            self.assertEqual(parent_end['error_type'], 'ValueError')
            self.assertGreaterEqual(parent_end['wall_s'], child_end['wall_s'])

    def test_unwritable_timing_does_not_turn_success_into_failure(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            with span(Path(folder), 'complete'):
                value = 42
            self.assertEqual(value, 42)



if __name__ == '__main__':
    unittest.main()
