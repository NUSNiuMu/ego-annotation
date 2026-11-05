"""Behavioral invariants for review acceptance, not generated wording."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ego_annotation.review import ROOT, digest, finalize, pin, validate, prepare_images
from ego_annotation.review_transport import output_schema


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.packet = {
            'input_files': [], 'source_frames': 8, 'binding': {'source_id': 'test'},
            'sample_frames': {str(i): {'start_ns': i * 500000000, 'next_ns': (i + 1) * 500000000}
                              for i in range(8)},
            'policy': json.loads((ROOT / 'skills/garment-review-policy.json').read_text()),
            'candidates': [{'id': 'w000:s00', 'window_frames': list(range(8))}],
        }
        self.packet['packet_id'] = digest(self.packet)
        self.result = {
            'packet_id': self.packet['packet_id'], 'review_type': 'text',
            'episodes': [{'id': 'episode-001', 'start_frame': 0, 'end_frame_exclusive': 8,
                          'disposition': 'ACCEPT', 'reason': 'Supported complete task.',
                          'boundary_notes': '', 'evidence_refs': ['w000:s00'], 'issues': [], 'unlabeled': [],
                          'actions': [{'code': 'smooth_garment', 'start_frame': 0, 'end_frame_exclusive': 8,
                                       'description': 'Smooth the garment.', 'source_refs': ['w000:s00'],
                                       'evidence_frames': [0, 3, 7], 'edit': 'normalize', 'reason': 'Explicit source description.'}]}],
            'excluded': [],
        }

    def test_valid_text_result_and_wrong_source_rejected(self):
        self.assertEqual(validate(self.packet, self.result)['accepted'], 1)
        self.result['packet_id'] = 'unrelated-source'
        with self.assertRaisesRegex(ValueError, 'binding'):
            validate(self.packet, self.result)

    def test_unseen_evidence_and_vocabulary_are_rejected(self):
        other = deepcopy(self.result)
        other['episodes'][0]['actions'][0]['evidence_frames'] = [100]
        with self.assertRaisesRegex(ValueError, 'Evidence'):
            validate(self.packet, other)
        self.result['episodes'][0]['actions'][0]['code'] = 'invented_action'
        with self.assertRaisesRegex(ValueError, 'Unknown action'):
            validate(self.packet, self.result)

    def test_nonpositive_source_duration_is_rejected(self):
        self.packet['sample_frames']['7']['next_ns'] = -1
        self.packet['packet_id'] = digest({k: v for k, v in self.packet.items() if k != 'packet_id'})
        self.result['packet_id'] = self.packet['packet_id']
        with self.assertRaisesRegex(ValueError, 'source clock'):
            validate(self.packet, self.result)

    def test_unresolved_phone_isolated_without_discarding_other_episode(self):
        episode = self.result['episodes'][0]
        episode['disposition'] = 'REVIEW'
        episode['issues'] = [{'id': 'phone', 'kind': 'PHONE_INTERRUPTION',
                              'question': 'Observed phone operation interrupts the task.',
                              'start_frame': 2, 'end_frame_exclusive': 4,
                              'source_refs': ['w000:s00'], 'requested_frames': [2, 3]}]
        self.assertEqual(validate(self.packet, self.result)['accepted'], 0)
        episode['disposition'] = 'ACCEPT'
        with self.assertRaises(ValueError):
            validate(self.packet, self.result)

    def test_issue_context_can_extend_beyond_episode_but_not_source(self):
        e = self.result['episodes'][0]
        e.update(start_frame=2, end_frame_exclusive=6, disposition='REVIEW')
        e['actions'][0].update(start_frame=2, end_frame_exclusive=6, evidence_frames=[2, 3, 5])
        e['issues'] = [{'id': 'end-context', 'kind': 'COMPLETION', 'question': 'Is the end static holding?',
                        'start_frame': 0, 'end_frame_exclusive': 8,
                        'source_refs': ['w000:s00'], 'requested_frames': [0, 2, 7]}]
        self.assertEqual(validate(self.packet, self.result)['issues'], 1)
        e['issues'][0]['end_frame_exclusive'] = 9
        with self.assertRaisesRegex(ValueError, 'source interval'):
            validate(self.packet, self.result)
        e['issues'][0]['end_frame_exclusive'] = 8
        e['actions'][0]['start_frame'] = 0
        with self.assertRaisesRegex(ValueError, 'outside episode'):
            validate(self.packet, self.result)

    def test_context_does_not_add_unrelated_neighbor_to_image_review(self):
        first = self.result['episodes'][0]
        first.update(end_frame_exclusive=4, disposition='REVIEW')
        first['actions'][0].update(end_frame_exclusive=4, evidence_frames=[0, 3])
        neighbor = deepcopy(first)
        neighbor.update(id='episode-002', start_frame=4, end_frame_exclusive=8, disposition='ACCEPT')
        neighbor['actions'][0].update(start_frame=4, end_frame_exclusive=8, evidence_frames=[4, 7])
        self.result['episodes'].append(neighbor)
        first['issues'] = [{'id': 'identity', 'kind': 'GARMENT_IDENTITY', 'question': 'Was the item exchanged?',
                            'start_frame': 0, 'end_frame_exclusive': 8,
                            'source_refs': ['w000:s00'], 'requested_frames': [0, 3, 7]}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); samples = []
            for frame in [0, 3, 7]:
                p = root / f'{frame}.jpg'; p.write_bytes(b'fixture')
                samples.append({'index': frame, 'relative_s': frame * 0.5,
                                'path': p.name, 'sha256': pin(p)['sha256']})
            (root/'samples.json').write_text(json.dumps(samples))
            self.packet.update(prepared=str(root), camera_topic='fixture')
            self.packet['packet_id'] = digest({k:v for k,v in self.packet.items() if k != 'packet_id'})
            self.result['packet_id'] = self.packet['packet_id']
            (root/'packet.json').write_text(json.dumps(self.packet))
            (root/'text.json').write_text(json.dumps(self.result))
            prepare_images(root/'packet.json', root/'text.json', root/'images.json')
            packet = json.loads((root/'images.json').read_text())
            self.assertEqual([e['id'] for e in packet['episodes']], ['episode-001'])
            self.assertEqual([i['frame'] for i in packet['images']], [0, 3, 7])

    def test_candidate_cannot_disappear_and_episodes_cannot_overlap(self):
        self.result['episodes'] = []
        with self.assertRaisesRegex(ValueError, 'accounted'):
            validate(self.packet, self.result)
        self.result['excluded'] = [{'source_ref': 'w000:s00', 'reason': 'Background.'}]
        self.assertEqual(validate(self.packet, self.result)['episodes'], 0)

    def image_fixture(self):
        issue = {'id': 'conflict', 'kind': 'ACTION_CONFLICT', 'question': 'Is the fold continuing?',
                 'start_frame': 2, 'end_frame_exclusive': 4,
                 'source_refs': ['w000:s00'], 'requested_frames': [2, 3]}
        packet = {'text_packet_id': self.packet['packet_id'], 'input_files': [],
                  'episodes': deepcopy(self.result['episodes']), 'issues': [issue],
                  'images': [{'frame': 2}, {'frame': 3}]}
        packet['episodes'][0].update(disposition='REVIEW',issues=[issue])
        packet['packet_id'] = digest(packet)
        result = deepcopy(self.result)
        result.update(packet_id=packet['packet_id'], review_type='image',
                      issue_results=[{'issue_id': 'conflict', 'resolution': 'RESOLVED',
                                      'observation': 'Compatible continued fold is visible.',
                                      'evidence_frames': [2, 3]}])
        result['episodes'][0]['source_episode_ids']=['episode-001']
        return packet, result, issue

    def test_image_result_bound_to_exact_affected_episode_and_images(self):
        packet, result, _ = self.image_fixture()
        self.assertEqual(validate(self.packet, result, image_packet=packet)['accepted'], 1)
        result['issue_results'][0]['evidence_frames'] = [6]
        with self.assertRaisesRegex(ValueError, 'image evidence'):
            validate(self.packet, result, image_packet=packet)
        result['issue_results'][0]['evidence_frames'] = [2]
        result['episodes'][0]['source_episode_ids'] = ['unrelated-parent']
        with self.assertRaisesRegex(ValueError, 'identities'):
            validate(self.packet, result, image_packet=packet)

    def test_unresolved_image_question_cannot_be_silently_removed(self):
        packet, result, issue = self.image_fixture()
        result['issue_results'][0]['resolution'] = 'UNRESOLVED'
        with self.assertRaisesRegex(ValueError, 'remain attached'):
            validate(self.packet, result, image_packet=packet)
        result['episodes'][0]['disposition'] = 'REVIEW'
        result['episodes'][0]['issues'] = [issue]
        self.assertEqual(validate(self.packet, result, image_packet=packet)['issues'], 1)

    def test_image_resolution_retires_pending_exclusion(self):
        image_packet, image_result, issue = self.image_fixture()
        self.result['episodes'][0]['disposition'] = 'REVIEW'
        self.result['episodes'][0]['issues'] = [issue]
        self.result['excluded'] = [{'source_ref': 'w000:s00', 'reason': 'Pending image issue.'}]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            pp, tp, ip, ir = [folder / name for name in ('packet.json', 'text.json', 'images.json', 'result.json')]
            pp.write_text(json.dumps(self.packet))
            tp.write_text(json.dumps(self.result))
            image_packet['input_files'] = [pin(tp)]
            image_packet['packet_id'] = digest({k: v for k, v in image_packet.items() if k != 'packet_id'})
            image_result['packet_id'] = image_packet['packet_id']
            ip.write_text(json.dumps(image_packet))
            ir.write_text(json.dumps(image_result))
            receipt = finalize(pp, tp, folder / 'final', image_packet_path=ip, image_result_path=ir)
            self.assertEqual(receipt['accepted_episodes'], 1)
            self.assertEqual(json.loads((folder / 'final/exclusions.json').read_text()), [])


class ReviewV2Tests(unittest.TestCase):
    def setUp(self):
        ReviewTests.setUp(self)

    def blank(self, start=0, end=8):
        return {'start_frame': start, 'end_frame_exclusive': end,
                'reason': 'The complete attempt is supported, but this internal action label is unclear.',
                'source_refs': ['w000:s00'], 'evidence_frames': [start, end - 1]}

    def finalize_result(self, folder):
        folder = Path(folder)
        (folder / 'packet.json').write_text(json.dumps(self.packet))
        (folder / 'text.json').write_text(json.dumps(self.result))
        return finalize(folder / 'packet.json', folder / 'text.json', folder / 'final')

    def test_short_action_keeps_source_time_without_padding(self):
        action = self.result['episodes'][0]['actions'][0]
        action.update(end_frame_exclusive=2, evidence_frames=[0, 1])
        self.assertEqual(validate(self.packet, self.result)['accepted'], 1)
        with tempfile.TemporaryDirectory() as folder:
            receipt = self.finalize_result(folder)
            saved = json.loads((Path(folder) / 'final/episodes.json').read_text())['episodes'][0]['actions'][0]
        self.assertEqual(saved['end_frame_exclusive'], 2)
        self.assertEqual(saved['end_s'] - saved['start_s'], 1.0)
        self.assertEqual(receipt['policy_version'], 'garment-review-v2')

    def test_unlabeled_only_complete_episode_preserves_interval_and_evidence(self):
        episode = self.result['episodes'][0]
        episode['actions'] = []
        episode['unlabeled'] = [self.blank()]
        self.assertEqual(validate(self.packet, self.result)['accepted'], 1)
        with tempfile.TemporaryDirectory() as folder:
            receipt = self.finalize_result(folder)
            saved = json.loads((Path(folder) / 'final/episodes.json').read_text())['episodes'][0]
        self.assertEqual(saved['unlabeled'][0]['start_s'], 0)
        self.assertEqual(saved['unlabeled'][0]['end_s'], 4)
        self.assertEqual(saved['unlabeled'][0]['source_refs'], ['w000:s00'])
        self.assertEqual(saved['unlabeled'][0]['evidence_frames'], [0, 7])
        self.assertEqual(receipt['actions'], 0)
        self.assertEqual(receipt['unlabeled_spans'], 1)

    def test_named_action_and_gap_must_not_overlap(self):
        episode = self.result['episodes'][0]
        episode['actions'][0].update(end_frame_exclusive=2, evidence_frames=[0, 1])
        episode['unlabeled'] = [self.blank(2, 8)]
        self.assertEqual(validate(self.packet, self.result)['accepted'], 1)
        episode['unlabeled'][0] = self.blank(1, 8)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate(self.packet, self.result)

    def test_gap_requires_reason_real_evidence_and_source(self):
        episode = self.result['episodes'][0]
        episode['actions'] = []
        for key, invalid in [('reason', ''), ('evidence_frames', []),
                             ('evidence_frames', [100]), ('source_refs', ['missing'])]:
            with self.subTest(field=key, value=invalid):
                span = self.blank(); span[key] = invalid
                episode['unlabeled'] = [span]
                with self.assertRaises(ValueError):
                    validate(self.packet, self.result)

    def test_unlabeled_cannot_hide_out_of_bounds_unordered_or_empty_task(self):
        episode = self.result['episodes'][0]
        episode['actions'] = []
        for spans in ([], [self.blank(0, 9)], [self.blank(4, 8), self.blank(0, 4)],
                      [self.blank(0, 5), self.blank(4, 8)], [self.blank(2, 2)]):
            with self.subTest(spans=spans):
                episode['unlabeled'] = spans
                with self.assertRaises(ValueError):
                    validate(self.packet, self.result)

    def test_complete_task_still_requires_resolved_episode_issues(self):
        episode = self.result['episodes'][0]
        episode['actions'] = []
        episode['unlabeled'] = [self.blank()]
        for kind in ('PHONE_INTERRUPTION', 'COMPLETION', 'TASK_BOUNDARY', 'GARMENT_IDENTITY'):
            with self.subTest(kind=kind):
                episode['issues'] = [{'id': 'unresolved', 'kind': kind, 'question': 'Task-level fact unclear.',
                                      'start_frame': 0, 'end_frame_exclusive': 8,
                                      'source_refs': ['w000:s00'], 'requested_frames': [0, 7]}]
                episode['disposition'] = 'ACCEPT'
                with self.assertRaises(ValueError):
                    validate(self.packet, self.result)
                episode['disposition'] = 'REVIEW'
                self.assertEqual(validate(self.packet, self.result)['accepted'], 0)

    def test_duration_alone_cannot_be_v2_issue(self):
        episode = self.result['episodes'][0]
        episode['disposition'] = 'REVIEW'
        episode['issues'] = [{'id': 'short', 'kind': 'SHORT_ACTION', 'question': 'Below 1.5 seconds.',
                              'start_frame': 0, 'end_frame_exclusive': 2,
                              'source_refs': ['w000:s00'], 'requested_frames': [0, 1]}]
        with self.assertRaisesRegex(ValueError, 'Invalid issue'):
            validate(self.packet, self.result)

    def test_image_result_can_resolve_label_question_into_documented_gap(self):
        episode = self.result['episodes'][0]
        episode['disposition'] = 'REVIEW'
        episode['actions'] = []
        issue = {'id': 'label', 'kind': 'ACTION_CONFLICT', 'question': 'Is task completion supported?',
                 'start_frame': 0, 'end_frame_exclusive': 8,
                 'source_refs': ['w000:s00'], 'requested_frames': [0, 7]}
        episode['issues'] = [issue]
        self.result['excluded'] = [{'source_ref': 'w000:s00', 'reason': 'Pending image issue.'}]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            pp, tp, ip, ir = [folder / n for n in ('packet.json', 'text.json', 'images.json', 'image.json')]
            pp.write_text(json.dumps(self.packet)); tp.write_text(json.dumps(self.result))
            image_packet = {'text_packet_id': self.packet['packet_id'], 'input_files': [pin(tp)],
                            'episodes': [episode], 'issues': [issue], 'images': [{'frame': 0}, {'frame': 7}]}
            image_packet['packet_id'] = digest(image_packet)
            result = deepcopy(self.result)
            result.update(packet_id=image_packet['packet_id'], review_type='image', excluded=[],
                          issue_results=[{'issue_id': 'label', 'resolution': 'RESOLVED',
                                          'observation': 'Completion is supported; label remains explicitly blank.',
                                          'evidence_frames': [0, 7]}])
            result['episodes'][0].update(disposition='ACCEPT', issues=[], unlabeled=[self.blank()], source_episode_ids=['episode-001'])
            ip.write_text(json.dumps(image_packet)); ir.write_text(json.dumps(result))
            receipt = finalize(pp, tp, folder / 'final', image_packet_path=ip, image_result_path=ir)
            self.assertEqual(receipt['accepted_episodes'], 1)
            self.assertEqual(receipt['unlabeled_spans'], 1)
            self.assertEqual(json.loads((folder / 'final/exclusions.json').read_text()), [])

    def test_schema_requires_explicit_gaps_for_both_stages(self):
        for stage in ('text', 'image'):
            current = output_schema(stage, self.packet['packet_id'])
            self.assertIn('unlabeled', current['properties']['episodes']['items']['required'])
            self.assertNotIn('SHORT_ACTION', current['properties']['episodes']['items']['properties']['issues']['items']['properties']['kind']['enum'])

    def test_superseded_policy_is_rejected_instead_of_compatibly_validated(self):
        self.packet['policy']['version'] = 'superseded-policy'
        self.packet['packet_id'] = digest({k: v for k, v in self.packet.items() if k != 'packet_id'})
        self.result['packet_id'] = self.packet['packet_id']
        with self.assertRaisesRegex(ValueError, 'Unsupported review policy'):
            validate(self.packet, self.result)

    def test_current_result_cannot_omit_unlabeled_field(self):
        del self.result['episodes'][0]['unlabeled']
        with self.assertRaisesRegex(ValueError, 'Unexpected fields'):
            validate(self.packet, self.result)



class ReviewIsolationTests(unittest.TestCase):
    def setUp(self):
        ReviewTests.setUp(self)
        first=self.result['episodes'][0]
        first['end_frame_exclusive']=4
        first['actions'][0].update(end_frame_exclusive=4,evidence_frames=[0,3])
        second=deepcopy(first);second.update(id='episode-002',start_frame=4,end_frame_exclusive=8)
        second['actions'][0].update(start_frame=4,end_frame_exclusive=8,evidence_frames=[4,7])
        self.result['episodes'].append(second)

    def run_final(self, result=None, ip=None, ir=None, image_error=None):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pp=root/'packet.json';tp=root/'text.json'
            pp.write_text(json.dumps(self.packet));tp.write_text(json.dumps(result or self.result))
            kwargs={}
            if ip:
                ip=deepcopy(ip);ip['input_files']=[pin(tp)]
                ip['packet_id']=digest({k:v for k,v in ip.items() if k!='packet_id'})
                (root/'ip.json').write_text(json.dumps(ip))
                kwargs={'image_packet_path':root/'ip.json','image_error':image_error}
                if ir:
                    ir=deepcopy(ir);ir['packet_id']=ip['packet_id']
                    (root/'ir.json').write_text(json.dumps(ir));kwargs['image_result_path']=root/'ir.json'
            receipt=finalize(pp,tp,root/'final',**kwargs)
            return receipt,json.loads((root/'final/episodes.json').read_text()),json.loads((root/'final/deferred.json').read_text())

    def pending_image(self):
        first=self.result['episodes'][0];first['disposition']='REVIEW'
        issue={'id':'cut','kind':'TASK_BOUNDARY','question':'Are these independent attempts?',
               'start_frame':0,'end_frame_exclusive':4,'source_refs':['w000:s00'],'requested_frames':[0,2,3]}
        first['issues']=[issue]
        ip={'text_packet_id':self.packet['packet_id'],'input_files':[],
            'episodes':[deepcopy(first)],'issues':[issue],'images':[{'frame':0},{'frame':2},{'frame':3}]}
        ip['packet_id']=digest(ip)
        child=deepcopy(first);child.update(disposition='ACCEPT',issues=[],source_episode_ids=['episode-001'])
        ir={'packet_id':ip['packet_id'],'review_type':'image','episodes':[child],'excluded':[],
            'issue_results':[{'issue_id':'cut','resolution':'RESOLVED','observation':'Visible independent attempts.', 'evidence_frames':[0,3]}]}
        return ip,ir

    def test_local_bad_action_keeps_other_episode_and_exact_source_scope(self):
        self.result['episodes'][0]['actions'][0]['evidence_frames']=[100]
        receipt,accepted,pending=self.run_final()
        self.assertEqual(receipt['accepted_episodes'],1)
        self.assertEqual(accepted['episodes'][0]['start_frame'],4)
        self.assertEqual(receipt['deferred_episodes'],1)
        self.assertEqual(pending['execution_issues'][0]['start_frame'],0)
        self.assertEqual(pending['execution_issues'][0]['end_frame_exclusive'],4)

    def test_conflicting_pair_is_withheld_without_discarding_disjoint_neighbor(self):
        third=deepcopy(self.result['episodes'][1]);third['id']='overlapping'
        self.result['episodes'].append(third)
        receipt,accepted,pending=self.run_final()
        self.assertEqual(receipt['accepted_episodes'],1)
        self.assertEqual(accepted['episodes'][0]['start_frame'],0)
        self.assertEqual({e['episode_id'] for e in pending['execution_issues']},{'episode-002','overlapping'})

    def test_wrong_shared_binding_still_blocks_every_episode(self):
        self.result['packet_id']='wrong-source'
        with self.assertRaisesRegex(ValueError,'binding'):self.run_final()

    def test_image_failure_keeps_unaffected_episode_without_accepting_pending_parent(self):
        ip,_=self.pending_image()
        receipt,accepted,pending=self.run_final(ip=ip,image_error='Transport retries exhausted')
        self.assertEqual(receipt['accepted_episodes'],1)
        self.assertEqual(accepted['episodes'][0]['source_episode_ids'],['episode-002'])
        self.assertEqual(pending['execution_issues'][0]['episode_id'],'episode-001')

    def test_one_parent_can_split_with_stable_host_ids(self):
        ip,ir=self.pending_image();first=ir['episodes'][0]
        first.update(id='child-A',end_frame_exclusive=2)
        first['actions'][0].update(end_frame_exclusive=2,evidence_frames=[0,1])
        second=deepcopy(first);second.update(id='child-B',start_frame=2,end_frame_exclusive=4)
        second['actions'][0].update(start_frame=2,end_frame_exclusive=4,evidence_frames=[2,3])
        ir['episodes'].append(second)
        receipt,accepted,pending=self.run_final(ip=ip,ir=ir)
        self.assertEqual(receipt['accepted_episodes'],3)
        self.assertEqual(pending['execution_issues'],[])
        old=[e['id'] for e in accepted['episodes']]
        ir['episodes'][0]['id']='different-local-name'
        again=self.run_final(ip=ip,ir=ir)[1]
        self.assertEqual([e['id'] for e in again['episodes']],old)

    def test_invalid_child_withholds_its_replacement_group_only(self):
        ip,ir=self.pending_image();ir['episodes'][0]['actions'][0]['evidence_frames']=[99]
        receipt,accepted,pending=self.run_final(ip=ip,ir=ir)
        self.assertEqual(receipt['accepted_episodes'],1)
        self.assertEqual(accepted['episodes'][0]['start_frame'],4)
        self.assertEqual(receipt['deferred_episodes'],1)

    def test_added_image_evidence_survives_merge_and_does_not_leak_to_neighbor(self):
        self.packet['candidates'][0]['window_frames']=[0,1,3,4,7]
        self.packet['packet_id']=digest({k:v for k,v in self.packet.items() if k!='packet_id'})
        self.result['packet_id']=self.packet['packet_id'];ip,ir=self.pending_image()
        ir['episodes'][0]['actions'][0]['evidence_frames']=[0,2,3]
        receipt,accepted,_=self.run_final(ip=ip,ir=ir)
        self.assertEqual(receipt['accepted_episodes'],2)
        self.assertIn(2,accepted['episodes'][0]['actions'][0]['evidence_frames'])
        self.result['episodes'][1]['actions'][0]['evidence_frames']=[5]
        receipt,accepted,pending=self.run_final(ip=ip,ir=ir)
        self.assertEqual(receipt['accepted_episodes'],1)
        self.assertEqual(accepted['episodes'][0]['start_frame'],0)
        self.assertTrue(pending['execution_issues'])

    def test_half_open_frame_hints_are_constrained_without_rewriting_original(self):
        from ego_annotation.review import constrain_requests
        ip,_=self.pending_image();self.result['episodes'][0]['issues'][0]['requested_frames']=[0,3,4,999]
        original=deepcopy(self.result);value,changes=constrain_requests(self.packet,self.result)
        self.assertEqual(self.result,original)
        self.assertEqual(value['episodes'][0]['issues'][0]['requested_frames'],[0,3])
        self.assertEqual(changes[0]['before'],[0,3,4,999])

    def test_named_affected_parents_can_merge_without_losing_lineage(self):
        ip,ir=self.pending_image();second=self.result['episodes'][1]
        question=deepcopy(ip['issues'][0]);question.update(id='second-cut',start_frame=4,end_frame_exclusive=8,requested_frames=[4,7])
        second.update(disposition='REVIEW',issues=[question]);ip['episodes'].append(deepcopy(second));ip['issues'].append(question)
        ip['images'] += [{'frame':4},{'frame':7}]
        ir['episodes'][0].update(end_frame_exclusive=8,source_episode_ids=['episode-001','episode-002'])
        ir['episodes'][0]['actions'][0].update(end_frame_exclusive=8,evidence_frames=[0,7])
        ir['issue_results'].append({'issue_id':'second-cut','resolution':'RESOLVED','observation':'One continuous attempt.', 'evidence_frames':[4,7]})
        receipt,accepted,pending=self.run_final(ip=ip,ir=ir)
        self.assertEqual(receipt['accepted_episodes'],1)
        self.assertEqual(accepted['episodes'][0]['source_episode_ids'],['episode-001','episode-002'])
        self.assertEqual(pending['execution_issues'],[])


if __name__ == '__main__':
    unittest.main()
