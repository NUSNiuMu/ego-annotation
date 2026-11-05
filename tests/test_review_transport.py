"""Exercise real subprocess/receipt plumbing; the executable is a test fixture."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from ego_annotation.review_transport import invoke, output_schema


class TransportTests(unittest.TestCase):
    def binary(self, root, *, tools=False, fail=False):
        path = root / 'fake-codex'
        path.write_text(f'''#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
schema = json.loads(pathlib.Path(args[args.index('--output-schema')+1]).read_text())
response = {{'packet_id': schema['properties']['packet_id']['enum'][0],
             'review_type': schema['properties']['review_type']['enum'][0],
             'episodes': [], 'excluded': []}}
pathlib.Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(response))
print(json.dumps({{'type':'thread.started','thread_id':'test-only'}}))
if {tools!r}:
    print(json.dumps({{'type':'item.completed','item':{{'type':'command_execution'}}}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':1,'output_tokens':1}}}}))
sys.stdin.read()
sys.exit({1 if fail else 0})
''')
        path.chmod(0o700)
        return str(path)

    def test_call_and_image_attachment_hash_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'test-only.jpg'
            image.write_bytes(b'test-only attachment')
            config = {'model': 'test-only', 'reasoning_effort': 'high',
                      'codex_binary': self.binary(root), 'call_timeout_seconds': 5}
            result, receipt = invoke(config, root / 'call', 'Test-only prompt.',
                                     output_schema('text', 'packet'), image_paths=[image])
            self.assertEqual(result['packet_id'], 'packet')
            self.assertEqual(receipt['returncode'], 0)
            self.assertEqual(receipt['tools_seen'], [])
            invocation = json.loads((root / 'call/invocation.json').read_text())
            self.assertEqual(invocation['attached_images'][0]['path'], str(image))
            self.assertEqual(receipt['thread_id'], 'test-only')

    def test_unexpected_tools_and_failure_are_preserved_and_rejected(self):
        for tools, fail in [(True, False), (False, True)]:
            with self.subTest(tools=tools, fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = {'model': 'test-only', 'reasoning_effort': 'high',
                          'codex_binary': self.binary(root, tools=tools, fail=fail), 'call_timeout_seconds': 5}
                with self.assertRaises(ValueError):
                    invoke(config, root / 'call', 'Test-only prompt.', output_schema('text', 'packet'))
                self.assertTrue((root / 'call/call-receipt.json').is_file())
                self.assertTrue((root / 'call/events.jsonl').is_file())
