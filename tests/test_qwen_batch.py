"""Independent real-MCAP checkpoints around a single fake batch model owner."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from PIL import Image

from ego_annotation import pipeline
from ego_annotation.config import PipelineConfig
from ego_annotation.qwen_batch import run_batched
from test_pipeline import FakeWorker, make_source


class BatchModel(FakeWorker):
    def __init__(self):
        super().__init__()
        self.batches = []
        self.threads = set()

    def infer_batch(self, requests):
        self.threads.add(threading.get_ident())
        self.batches.append(len(requests))
        return [self.infer(r['window'], r['samples'], r['images']) for r in requests]


class BatchTests(unittest.TestCase):
    def test_prefetched_take_is_admitted_after_short_handoff_while_long_runs(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d)
            sources=[make_source(root/'short'),make_source(root/'long',count=64),
                     make_source(root/'prefetched',count=16)]
            model=BatchModel();pending=iter(enumerate(sources));next_item=[next(pending)]
            completed=[];admitted=[];ready=threading.Event()
            def admit():
                item=next_item[0]
                if item is None or (item[0]==2 and not ready.is_set()):return None
                key,source=item;next_item[0]=next(pending,None)
                if key==2:
                    self.assertIn(0,completed)
                    self.assertNotIn(1,completed)
                admitted.append(key)
                return key,lambda proxy:pipeline.run_batch([[source]],root/str(key),
                    PipelineConfig(),lambda:proxy)[0]
            def finish(key,row):
                self.assertEqual(row['status'],'COMPLETE')
                completed.append(key)
                if key==0:ready.set()
            run_batched(admit,lambda:model,finish)
            self.assertEqual(admitted,[0,1,2])
            self.assertEqual(set(completed),{0,1,2})
            self.assertEqual(len(model.threads),1)

    def test_realistic_peer_decode_delay_still_forms_a_two_input_batch(self):
        model=BatchModel();started=threading.Event();pending=iter([0,1]);completed=[]
        def admit():
            key=next(pending,None)
            if key is None:return None
            def compute(proxy):
                if key:
                    started.wait();time.sleep(.08)
                else:started.set()
                return proxy.infer({'indices':[key]}, {key:{'relative_s':key}},
                                   [Image.new('RGB',(64,64))])
            return key,compute
        run_batched(admit,lambda:model,lambda key,value:completed.append(key))
        self.assertEqual(model.batches,[2])
        self.assertEqual(set(completed),{0,1})

    def test_short_take_finishes_independently_and_restart_keeps_completed_windows(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            sources = [make_source(root/'short'), make_source(root/'long', count=32)]
            model, completed = BatchModel(), []
            def run(interrupt):
                pending = iter(enumerate(sources))
                def admit():
                    item = next(pending, None)
                    if item is None:return None
                    key, source = item
                    return key, lambda proxy: pipeline.run_batch([[source]], root/str(key),
                        PipelineConfig(), lambda: proxy)[0]
                def finish(key, row):
                    completed.append((key, row))
                    if interrupt and key == 0:
                        raise KeyboardInterrupt('controlled owner interruption')
                run_batched(admit, lambda: model, finish)
            with self.assertRaises(KeyboardInterrupt):run(True)
            self.assertEqual(completed[0][0], 0)
            files = list(root.glob('*/jobs/*/windows/*.json'))
            self.assertGreaterEqual(len(files), 2)
            pins = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
            before = len(model.calls)
            run(False)
            self.assertEqual({k for k,r in completed if r['status']=='COMPLETE'}, {0,1})
            self.assertEqual(len(model.threads), 1)
            self.assertIn(2, model.batches)
            total = sum(len(list(p.glob('*.json'))) for p in root.glob('*/jobs/*/windows'))
            self.assertEqual(len(model.calls)-before, total-len(pins))
            for p, pin in pins.items():self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),pin)

    def test_model_failure_wakes_waiters_without_fallback_or_deadlock(self):
        def compute(proxy):
            return proxy.infer({'indices':[0]}, {}, [])
        items = iter([(1,compute),(2,compute)])
        def broken():raise RuntimeError('model error')
        with self.assertRaisesRegex(RuntimeError,'model error'):
            run_batched(lambda:next(items,None),broken,lambda *args:None)




if __name__=='__main__':unittest.main()
