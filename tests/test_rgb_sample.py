import ast, io, struct, unittest
from pathlib import Path
from unittest.mock import patch
from ego_annotation import sampling as r
from PIL import Image

class SamplingTests(unittest.TestCase):
    def test_tail_and_window_last(self):
        stamps=list(range(0,7_000_000_001,100_000_000))
        w=r.plan_windows(stamps,100_000_000,3,.5)
        self.assertEqual(len(w),3)
        self.assertEqual(w[0]['indices'],[0,5,10,15,20,25,29])
        self.assertEqual(w[-1]['indices'],[60,65,70])
        self.assertEqual(w[-1]['end_exclusive_ns'],7_100_000_000)
    def test_dedup_sparse(self):
        w=r.plan_windows([0,2_000_000_000],1_000_000_000,3,.25)
        self.assertEqual(w[0]['indices'],[0,1])
    def test_boundary_and_time_mapping(self):
        origin=6_000_000_000_003
        stamps=[origin+i*1_000_000_000 for i in range(7)]
        w=r.plan_windows(stamps,1_000_000_000,3,1)
        self.assertEqual(w[0]['indices'],[0,1,2]); self.assertEqual(w[1]['indices'],[3,4,5])
        for win in w:
            for t in win['targets']:
                self.assertEqual(t['actual_stamp_ns'],stamps[t['index']])
                self.assertGreaterEqual(t['actual_stamp_ns'],t['target_ns'])
    def test_no_upscale(self):
        self.assertEqual(r.target_size(1088,1920,960),(544,960))
        self.assertEqual(r.target_size(1088,1920,640),(363,640))
        self.assertEqual(r.target_size(100,200,960),(100,200))
    def test_jpeg_interface_equivalence(self):
        import tempfile
        data=r.jpeg_bytes(Image.new('RGB',(32,32),(20,80,170)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.jpg';p.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                self.assertEqual(image.convert('RGB').tobytes(),r.consume_jpeg_path(p).tobytes())
    def test_preparation_preserves_pinned_jpeg_without_decoding_it(self):
        import tempfile, hashlib, json
        from test_pipeline import make_source
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=make_source(root/'source')
            with patch.object(r,'consume_jpeg_path',side_effect=AssertionError('Preparation decoded a saved JPEG')):
                result=r.run([source],'/egohalo/record/cam0/h264',root/'prepared')
            samples=json.loads((root/'prepared/samples.json').read_text())
            self.assertEqual(len(samples),result['sampled_images'])
            for sample in samples:
                path=root/'prepared'/sample['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),sample['sha256'])
                self.assertEqual(r.consume_jpeg_path(path).size,(sample['width'],sample['height']))
            self.assertEqual(result['saved_images'],len(samples))
    def test_cdr_stamp(self):
        data=bytearray(b'\x00\x01\x00\x00'+struct.pack('<iI',12,345))
        for text in [b'rgb\0',b'h264\0']:
            data.extend(b'\0'*((-(len(data)-4))%4));data.extend(struct.pack('<I',len(text))+text)
        data.extend(b'\0'*((-(len(data)-4))%4));data.extend(struct.pack('<I',4)+b'test')
        self.assertEqual(r.CDR(data).image(),(12_000_000_345,'rgb','h264',b'test'))
        self.assertEqual(r.CDR(data+b'\0').image()[3],b'test')
        with self.assertRaises(ValueError):r.CDR(data+b'bad').image()
    def test_runtime_has_no_old_path_coupling(self):
        root=Path(r.__file__).parent
        source=Path(r.__file__).read_text()
        for banned in ['robot-dataset-annotator','workflow_v1','sys.path','subprocess','importlib','symlink','/home/']:
            self.assertNotIn(banned,source)
        tree=ast.parse(source)
        imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.extend(a.name.split('.')[0] for a in n.names)
            elif isinstance(n,ast.ImportFrom):imports.append(n.module.split('.')[0])
        allowed={'argparse','bisect','hashlib','io','json','math','platform','statistics','struct','sys','time','collections','fractions','pathlib','av','mcap','PIL','re'}
        self.assertTrue(set(imports)<=allowed)
        self.assertFalse(any(p.is_symlink() for p in root.glob('*.py')))
if __name__=='__main__': unittest.main()
