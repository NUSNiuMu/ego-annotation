import json,sys,unittest
from pathlib import Path
from ego_annotation.inference.schema import validate,parse_response
class SchemaTests(unittest.TestCase):
 def response(self):
  return {'segments':[{'start_frame':0,'end_frame':15,'action':'smooth_shirt','description':'The wearer smooths the shirt on the table.','evidence_frames':[0,15],'continues':'to_next'}],'uncertainties':[]}
 def test_accepts_sparse_inclusive_range(self):
  x=self.response();self.assertEqual(validate(json.dumps(x),[0,15]),x)
 def test_rejects_unseen_evidence_and_end(self):
  x=self.response();x['segments'][0]['evidence_frames']=[5]
  with self.assertRaises(ValueError):validate(json.dumps(x),[0,15])
  x=self.response();x['segments'][0]['end_frame']=16
  with self.assertRaises(ValueError):validate(json.dumps(x),[0,15])
 def test_rejects_extra_metadata_and_bad_continues(self):
  x=self.response();x['confidence']=1
  with self.assertRaises(ValueError):validate(json.dumps(x),[0,15])
  x=self.response();x['segments'][0]['continues']='yes'
  with self.assertRaises(ValueError):validate(json.dumps(x),[0,15])
 def test_does_not_repair_json(self):
  with self.assertRaises(ValueError):validate('```json\n'+json.dumps(self.response())+'\n```',[0,15])
 def test_outer_wrapper_only(self):
  text='```json\n'+json.dumps(self.response())+'\n```'
  parsed,error,valid,strict,removed=parse_response(text,[0,15])
  self.assertEqual(parsed,self.response());self.assertIsNone(error)
  self.assertTrue(valid);self.assertFalse(strict);self.assertTrue(removed)
 def test_partial_json_stays_failed(self):
  parsed,error,valid,strict,removed=parse_response('```json\n{"segments": [\n```',[0,15])
  self.assertIsNone(parsed);self.assertIsNotNone(error);self.assertFalse(valid)
 def test_runtime_imports_do_not_reference_old_code(self):
  root=Path(__file__).resolve().parents[1]
  for p in (root/'src/ego_annotation/inference').glob('*.py'):
   text=p.read_text()
   for banned in ['workflow_v1','robot_dataset_annotator','/opt/rda-workflow','/home/niumu/data']:
    self.assertNotIn(banned,text)
if __name__=='__main__':unittest.main()
