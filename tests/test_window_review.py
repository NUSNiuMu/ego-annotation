"""Faulty windows force bounded images; independent episodes remain usable."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ego_annotation import pipeline, review
from ego_annotation.config import PipelineConfig
from test_pipeline import FakeWorker, make_source
import test_review


class WindowReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = test_review.ReviewTests(); fixture.setUp()
        self.packet = fixture.packet
        self.packet['candidates'] = [
            {'id':'w000:gap','window':0,'start_frame':0,'end_frame_exclusive':4,
             'window_frames':[0,1,2,3],'context_frames':list(range(8)),
             'requires_image':True,'error':'Invalid evidence frames','raw_response':'unvalidated'},
            {'id':'w001:s00','window_frames':[4,5,6,7]}]
        self.packet.pop('packet_id'); self.packet['packet_id'] = review.digest(self.packet)
        self.text = deepcopy(fixture.result)
        self.text['packet_id'] = self.packet['packet_id']
        e = self.text['episodes'][0]; e['start_frame'] = 4
        e['evidence_refs'] = ['w001:s00']; e['actions'][0].update(
            start_frame=4,source_refs=['w001:s00'],evidence_frames=[4,7])
        self.text['excluded'] = [{'source_ref':'w000:gap','reason':'Model guesses idle.'}]

    def image_result(self, *, resolution='RESOLVED', exclude=False):
        assessment = review.assess(self.packet,self.text)
        parent = next(e for e in assessment['episodes'] if e['id'].startswith('qwen-gap'))
        issue = parent['issues'][0]
        ip = {'text_packet_id':self.packet['packet_id'],'input_files':[],
              'episodes':[parent],'issues':[issue],'images':[{'frame':i} for i in range(8)]}
        ip['packet_id'] = review.digest(ip)
        child = deepcopy(parent); child['id'] = 'fixed'; child['source_episode_ids'] = [parent['id']]
        if resolution == 'RESOLVED':
            child['disposition'] = 'ACCEPT'; child['issues'] = []
            child['unlabeled'] = [{'start_frame':0,'end_frame_exclusive':4,'reason':'Visible complete task, action unclear.',
                'source_refs':['w000:gap'],'evidence_frames':[0,3]}]
        ir = {'packet_id':ip['packet_id'],'review_type':'image','episodes':[] if exclude else [child],
              'excluded':[{'source_ref':'w000:gap','reason':'Images show idle.'}] if exclude else [],
              'issue_results':[{'issue_id':issue['id'],'resolution':resolution,
                                'observation':'Actual attached frames inspected.','evidence_frames':[0,3]}]}
        return ip,ir

    def test_text_cannot_hide_gap_as_exclusion_or_accept_it(self):
        assessment = review.assess(self.packet,self.text)
        self.assertEqual(assessment['validation']['accepted'],1)
        self.assertEqual(assessment['excluded'],[])
        gap = next(e for e in assessment['episodes'] if e['disposition']=='REVIEW')
        self.assertEqual(gap['issues'][0]['requested_frames'],list(range(8)))
        self.assertEqual(self.text['excluded'][0]['source_ref'],'w000:gap')
        full = deepcopy(self.text); e = full['episodes'][0]; e['start_frame']=0
        self.assertEqual(review.assess(self.packet,full)['validation']['accepted'],0)

    def test_images_resolve_or_defer_only_affected_scope(self):
        for resolution in ('RESOLVED','UNRESOLVED'):
            with self.subTest(resolution=resolution):
                ip,ir = self.image_result(resolution=resolution)
                assessment = review.assess(self.packet,ir,image_packet=ip)
                self.assertEqual(assessment['execution_issues'],[])
                self.assertEqual(assessment['validation']['accepted'],int(resolution=='RESOLVED'))

    def test_neighbor_only_evidence_cannot_resolve_bad_window(self):
        ip,ir = self.image_result(); ir['issue_results'][0]['evidence_frames']=[7]
        assessment = review.assess(self.packet,ir,image_packet=ip)
        self.assertEqual(assessment['episodes'],[])
        self.assertTrue(assessment['execution_issues'])

    def test_finalization_accepts_repaired_task_or_verified_idle(self):
        for exclude in (False,True):
            with self.subTest(exclude=exclude), tempfile.TemporaryDirectory() as d:
                root = Path(d); packet = root/'packet.json'; text = root/'text.json'
                review.write_new(packet,self.packet); review.write_new(text,self.text)
                ip,ir = self.image_result(exclude=exclude)
                ip['input_files'] = [review.pin(text)]; ip.pop('packet_id'); ip['packet_id']=review.digest(ip)
                ir['packet_id']=ip['packet_id']
                review.write_new(root/'image-packet.json',ip); review.write_new(root/'image.json',ir)
                receipt = review.finalize(packet,text,root/'final',image_packet_path=root/'image-packet.json',
                                         image_result_path=root/'image.json')
                self.assertEqual(receipt['accepted_episodes'],1 if exclude else 2)
                self.assertEqual(receipt['deferred_episodes'],0)

    def test_failed_images_keep_unrelated_episode(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); review.write_new(root/'packet.json',self.packet); review.write_new(root/'text.json',self.text)
            ip,_ = self.image_result(); ip['input_files']=[review.pin(root/'text.json')]
            ip.pop('packet_id'); ip['packet_id']=review.digest(ip)
            review.write_new(root/'image-packet.json',ip)
            receipt = review.finalize(root/'packet.json',root/'text.json',root/'final',
                                     image_packet_path=root/'image-packet.json',image_error='transport failed')
            self.assertEqual(receipt['accepted_episodes'],1)
            self.assertGreater(receipt['deferred_episodes'],0)
