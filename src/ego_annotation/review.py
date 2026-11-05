"""Prepare and validate skill reviews without invoking models or touching Qwen jobs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
METHOD_FILES = ('garment-review-policy.json', 'review-contract.md',
                'ego-text-review/SKILL.md', 'ego-image-review/SKILL.md')
ISSUE_KINDS = {'TASK_BOUNDARY', 'GARMENT_IDENTITY', 'COMPLETION', 'ACTION_CONFLICT',
               'MISSING_CONTENT', 'PHONE_INTERRUPTION'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def fields(value, expected):
    check(isinstance(value, dict) and set(value) == set(expected.split()),
          'Unexpected fields: ' + str(set(value) if isinstance(value, dict) else type(value)))


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def verify_packet(packet):
    body = {k: v for k, v in packet.items() if k != 'packet_id'}
    check(digest(body) == packet['packet_id'], 'Packet identity mismatch')
    for item in packet['input_files']:
        check(pin(item['path']) == item, 'Changed input: ' + item['path'])
    return packet


def prepare(job_dir, output, *, prepared_dir=None, method_dir=None):
    job_dir = Path(job_dir).resolve()
    job, status = read(job_dir / 'job.json'), read(job_dir / 'status.json')
    check(status['status'] == 'COMPLETE', 'Qwen job must be complete')
    prepared = (Path(prepared_dir).resolve() if prepared_dir else
                job_dir.parent.parent / 'sources' / job['binding']['source_id'] / Path(job['prepared']).name)
    method_dir = Path(method_dir).resolve() if method_dir else ROOT / 'skills'
    policy = read(method_dir / 'garment-review-policy.json')
    check(policy.get('version') == 'garment-review-v2', 'Unsupported review policy')
    frames, samples = read(prepared / 'all_frames.json'), read(prepared / 'samples.json')
    source, windows = read(prepared / 'source.json'), read(prepared / 'windows.json')
    check(all(f['index'] == i for i, f in enumerate(frames)), 'Noncanonical source frame map')
    stamps = [f['stamp_ns'] for f in frames]
    check(all(a < b for a, b in zip(stamps, stamps[1:])), 'Source clock is not increasing')
    stamps.append(stamps[-1] + source['median_period_ns'])
    first = stamps[0]
    paths = [job_dir / 'job.json', job_dir / 'status.json']
    paths += [prepared / n for n in ('all_frames.json', 'samples.json', 'source.json', 'windows.json')]
    paths += [method_dir / n for n in METHOD_FILES]
    candidates, uncertainties = [], []
    from .pipeline import checked_result
    for ordinal, window in enumerate(windows):
        path = job_dir / 'windows' / f'{ordinal:06d}.json'
        item = checked_result(path, job['binding'], window)
        paths.append(path)
        if 'reused_from' in item:
            paths += [job_dir/'reused'/path.name, job_dir/'reused/job.json']
        if item['status'] == 'NO_INPUT' and not window['indices']:
            check(item['parsed'] == {'segments': [], 'uncertainties': []}, 'Invalid empty Qwen window')
            continue
        if item['status'] != 'PASS':
            from .inference.schema import parse_response
            indices = window['indices']
            context = sorted({f for w in windows[max(0,ordinal-1):ordinal+2] for f in w['indices']})
            candidates.append({'id': f'w{ordinal:03d}:gap', 'window': ordinal,
                'start_frame': min(indices), 'end_frame_exclusive': max(indices)+1,
                'window_frames': indices, 'context_frames': context, 'requires_image': True,
                'error': parse_response(item['raw_response'], indices)[1] or item['status'],
                'raw_response': item['raw_response']})
            continue
        # Revalidate saved parsed results against immutable raw model text.
        from .inference.schema import parse_response
        parsed, error, *_ = parse_response(item['raw_response'], window['indices'])
        check(error is None and parsed == item['parsed'], 'Raw/parsed model result mismatch')
        for si, segment in enumerate(parsed['segments']):
            candidates.append({'id': f'w{ordinal:03d}:s{si:02d}', 'window': ordinal,
                               **segment, 'end_frame_exclusive': segment['end_frame'] + 1,
                               'start_s': (stamps[segment['start_frame']] - first) / 1e9,
                               'end_s': (stamps[segment['end_frame'] + 1] - first) / 1e9,
                               'window_frames': window['indices']})
        uncertainties.extend({'window': ordinal, **u} for u in parsed['uncertainties'])
    sample_frames = {str(s['index']): {'start_ns': stamps[s['index']] - first,
                                      'next_ns': stamps[s['index'] + 1] - first}
                     for s in samples}
    packet = {'kind': 'garment-text-review-input-v2', 'binding': job['binding'],
              'policy': policy,
              'input_files': [pin(p) for p in paths], 'source_frames': len(frames),
              'duration_ns': stamps[-1] - first, 'camera_topic': job['config']['topic'],
              'prepared': str(prepared), 'sample_frames': sample_frames,
              'candidates': candidates, 'uncertainties': uncertainties,
              'boundary_semantics': 'All review intervals half-open in original source frame coordinates; sparse cuts remain coarse.'}
    packet['packet_id'] = digest(packet)
    write_new(output, packet)
    return {'packet_id': packet['packet_id'], 'windows': len(windows), 'candidates': len(candidates)}


def require_window_images(packet, result, resolved_windows):
    """Host-owned issues cannot be skipped, excluded or accepted by text alone."""
    result = deepcopy(result)
    changes = []
    for candidate in packet['candidates']:
        if not candidate.get('requires_image') or candidate['id'] in resolved_windows:
            continue
        ref = candidate['id']; a, b = candidate['start_frame'], candidate['end_frame_exclusive']
        affected = [e for e in result['episodes'] if isinstance(e,dict)
                    and type(e.get('start_frame')) is int and type(e.get('end_frame_exclusive')) is int
                    and e['start_frame'] < b and a < e['end_frame_exclusive']]
        if not affected:
            episode = {'id': 'qwen-gap-'+str(candidate['window']), 'start_frame': a, 'end_frame_exclusive': b,
                'disposition': 'REVIEW', 'reason': 'Unvalidated Qwen window; task content needs images.',
                'boundary_notes': '', 'evidence_refs': [ref], 'actions': [], 'unlabeled': [], 'issues': []}
            result['episodes'].append(episode); affected = [episode]
        for episode in affected:
            if not isinstance(episode.get('issues'),list) or not isinstance(episode.get('evidence_refs'),list):
                continue  # Ordinary episode validation isolates malformed model fields.
            hints = candidate['context_frames']
            issue = {'id': f'qwen-window-{candidate["window"]}-{digest(episode["id"])[:8]}',
                'kind': 'MISSING_CONTENT', 'question': 'Resolve this unvalidated Qwen window from images: '+candidate['error'],
                'start_frame': min(hints), 'end_frame_exclusive': max(hints)+1,
                'source_refs': [ref], 'requested_frames': hints}
            episode['issues'] = [i for i in episode['issues'] if not isinstance(i,dict) or i.get('id') != issue['id']] + [issue]
            episode['disposition'] = 'REVIEW'
            if ref not in episode['evidence_refs']: episode['evidence_refs'].append(ref)
            changes.append({'episode_id':episode['id'], 'issue_id':issue['id'],
                            'reason':'Mandatory image review of unvalidated Qwen window', 'source_ref':ref})
        result['excluded'] = [e for e in result['excluded'] if e['source_ref'] != ref]
    return result, changes


def clock(packet, index):
    samples = packet['sample_frames']
    if str(index) in samples:
        return samples[str(index)]['start_ns']
    check(str(index - 1) in samples, 'Boundary must use a supplied sample or sample + 1')
    return samples[str(index - 1)]['next_ns']


def interval(packet, item):
    start, end = item['start_frame'], item['end_frame_exclusive']
    check(type(start) is int and type(end) is int and 0 <= start < end <= packet['source_frames'],
          'Invalid source interval')
    check(str(start) in packet['sample_frames'], 'Start must be a supplied sample')
    check(clock(packet, end) > clock(packet, start), 'Invalid source clock interval')
    return start, end


def refs(value, candidates):
    check(isinstance(value, list) and value and len(value) == len(set(value))
          and all(isinstance(v, str) and v in candidates for v in value), 'Invalid source references')


def span_evidence(span, a, b, candidates, image_packet=None):
    refs(span['source_refs'], candidates)
    supplied = {f for ref in span['source_refs'] for f in candidates[ref]['window_frames']}
    if image_packet is not None:
        supplied |= {f['frame'] for f in image_packet['images']}
    evidence = span['evidence_frames']
    check(isinstance(evidence, list) and evidence and all(
        type(f) is int and a <= f < b and f in supplied for f in evidence),
        'Evidence must be supplied frames inside span')


def episode_validation(packet, episode, candidates, image_packet=None):
    used, issue_ids = set(), set()
    fields(episode, 'id start_frame end_frame_exclusive disposition reason boundary_notes evidence_refs actions issues unlabeled')
    check(nonempty(episode['id']), 'Invalid episode ID')
    start, end = interval(packet, episode)
    check(episode['disposition'] in {'ACCEPT', 'REVIEW', 'INCOMPLETE'}, 'Invalid disposition')
    check(nonempty(episode['reason']) and isinstance(episode['boundary_notes'], str), 'Missing finding')
    refs(episode['evidence_refs'], candidates)
    check(isinstance(episode['actions'], list) and isinstance(episode['issues'], list), 'Expected episode lists')
    unlabeled = episode['unlabeled']
    check(isinstance(unlabeled, list), 'Expected unlabeled list')
    check(episode['disposition'] != 'ACCEPT' or ((episode['actions'] or unlabeled) and not episode['issues']),
          'Accepted episode must have actions or explicit unlabeled spans and no unresolved issues')
    action_end = start
    spans = []
    for action in episode['actions']:
        fields(action, 'code start_frame end_frame_exclusive description source_refs evidence_frames edit reason')
        a, b = interval(packet, action)
        check(start <= a < b <= end and a >= action_end, 'Action outside episode or overlapping')
        action_end = b
        spans.append((a, b))
        check(action['code'] in packet['policy']['actions'], 'Unknown action code')
        check(nonempty(action['description']) and nonempty(action['reason']), 'Missing action description/grounding')
        check(action['edit'] in {'retain', 'normalize', 'merge', 'split', 'supplement'}, 'Unknown edit')
        span_evidence(action, a, b, candidates, image_packet)
        used.update(action['source_refs'])
    unlabeled_end = start
    for span in unlabeled:
        fields(span, 'start_frame end_frame_exclusive reason source_refs evidence_frames')
        a, b = interval(packet, span)
        check(start <= a < b <= end and a >= unlabeled_end,
              'Unlabeled span outside episode or overlapping/unordered')
        unlabeled_end = b
        spans.append((a, b))
        check(nonempty(span['reason']), 'Missing unlabeled reason')
        span_evidence(span, a, b, candidates, image_packet)
        used.update(span['source_refs'])
    spans.sort()
    check(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])),
          'Actions and unlabeled spans overlap')
    for issue in episode['issues']:
        fields(issue, 'id kind question start_frame end_frame_exclusive source_refs requested_frames')
        check(nonempty(issue['id']) and issue['id'] not in issue_ids, 'Duplicate issue ID')
        issue_ids.add(issue['id'])
        a, b = interval(packet, issue)
        check(a < end and start < b and issue['kind'] in ISSUE_KINDS and nonempty(issue['question']),
              'Invalid issue: context must overlap its episode and use a valid kind/question')
        refs(issue['source_refs'], candidates)
        check(isinstance(issue['requested_frames'], list) and all(type(f) is int and a <= f < b and str(f) in packet['sample_frames'] for f in issue['requested_frames']), 'Invalid requested image frame')
        check(issue['kind'] != 'PHONE_INTERRUPTION' or episode['disposition'] == 'REVIEW', 'Phone-interrupted task cannot pass')
        used.update(issue['source_refs'])
    return used, issue_ids


def referenced(value, candidates):
    if isinstance(value, dict):
        return {r for k, v in value.items() if k in {'source_refs', 'evidence_refs'}
                and isinstance(v, list) for r in v if isinstance(r, str) and r in candidates} | set().union(
                    *(referenced(v, candidates) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(referenced(v, candidates) for v in value))
    return set()


def blocked(packet, value, reason, candidates):
    refs = sorted(referenced(value, candidates))
    try:
        start, end = interval(packet, value)
    except (ValueError, KeyError, TypeError):
        anchors = [f for r in refs for f in candidates[r]['window_frames']]
        start, end = (min(anchors), max(anchors)+1) if anchors else (0, packet['source_frames'])
    return {'episode_id': value.get('id') if isinstance(value, dict) else None,
            'start_frame': start, 'end_frame_exclusive': end, 'reason': reason, 'source_refs': refs}


def assess(packet, result, *, image_packet=None, image_evidence=None, reserved_refs=(), resolved_windows=()):
    """Shared identity failures raise; local failures quarantine only their episode/group."""
    verify_packet(packet)
    check(packet['policy'].get('version') == 'garment-review-v2', 'Unsupported review policy')
    image = image_packet is not None
    fields(result, 'packet_id review_type episodes excluded' + (' issue_results' if image else ''))
    if image:
        verify_packet(image_packet)
        check(image_packet['text_packet_id'] == packet['packet_id'], 'Image/text identity mismatch')
    check(result['packet_id'] == (image_packet or packet)['packet_id'], 'Result/input binding mismatch')
    check(result['review_type'] == ('image' if image else 'text'), 'Wrong review stage')
    check(isinstance(result['episodes'], list) and isinstance(result['excluded'], list), 'Expected lists')
    candidates = {c['id']: c for c in packet['candidates']}
    window_changes = []
    if not image:
        result, window_changes = require_window_images(packet, result, set(resolved_windows)|set(reserved_refs))
    excluded = set()
    for item in result['excluded']:
        fields(item, 'source_ref reason')
        check(item['source_ref'] in candidates and item['source_ref'] not in excluded
              and nonempty(item['reason']), 'Invalid exclusion')
        excluded.add(item['source_ref'])
    good, failures, used, issues = [], [], set(reserved_refs), set()
    parents = {e['id']: e for e in image_packet['episodes']} if image else {}
    bad_parents, mapped = set(), set()
    ids = [e.get('id') for e in result['episodes'] if isinstance(e, dict)]
    for original in result['episodes']:
        value = deepcopy(original)
        source_ids = []
        try:
            if image:
                source_ids = value.pop('source_episode_ids')
                check(isinstance(source_ids, list) and source_ids and len(source_ids)==len(set(source_ids))
                      and set(source_ids)<=set(parents), 'Invalid source episode identities')
                mapped.update(source_ids)
            check(ids.count(value.get('id')) == 1, 'Duplicate episode ID')
            if image:
                contexts=[parents[k] for k in source_ids]+[i for k in source_ids for i in parents[k]['issues']]
                check(min(c['start_frame'] for c in contexts)<=value['start_frame']<value['end_frame_exclusive']
                      <=max(c['end_frame_exclusive'] for c in contexts), 'Image replacement outside affected context')
            basis = image_packet if image else (image_evidence or {}).get(value['id'])
            refs, question_ids = episode_validation(packet, value, candidates, basis)
            check(not (issues & question_ids), 'Duplicate issue ID')
            issues.update(question_ids); used.update(refs); good.append(original)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            failures.append(blocked(packet, original, str(error), candidates))
            used.update(referenced(original, candidates))
            if image:
                bad_parents.update(set(source_ids) & set(parents) if isinstance(source_ids, list) else parents)
                if not source_ids: bad_parents.update(parents)
    # Conflicting episode ranges cannot be independently accepted.
    for i, left in enumerate(good):
        for right in good[i+1:]:
            if left['start_frame'] < right['end_frame_exclusive'] and right['start_frame'] < left['end_frame_exclusive']:
                for value in (left, right):
                    failures.append(blocked(packet,value,'Overlapping episode ranges',candidates))
                    if image: bad_parents.update(value['source_episode_ids'])
    if image:
        requested = {i['id']:i for i in image_packet['issues']}
        outcomes = result['issue_results']
        check(isinstance(outcomes,list), 'Expected issue results')
        for key, question in requested.items():
            affected = {e['id'] for e in parents.values() if any(i['id']==key for i in e['issues'])}
            check(affected, 'Image question has no affected parent')
            try:
                matches = [o for o in outcomes if o.get('issue_id')==key]
                check(len(matches)==1, 'Each image question needs one outcome')
                outcome = matches[0]
                fields(outcome, 'issue_id resolution observation evidence_frames')
                check(outcome['resolution'] in {'RESOLVED','UNRESOLVED'} and nonempty(outcome['observation']), 'Invalid image finding')
                frames = {i['frame'] for i in image_packet['images']}
                check(isinstance(outcome['evidence_frames'],list) and outcome['evidence_frames']
                      and all(type(f) is int and f in frames for f in outcome['evidence_frames']), 'Invalid image evidence')
                for ref in question['source_refs']:
                    c = candidates[ref]
                    if c.get('requires_image'):
                        check(any(f in c['window_frames'] for f in outcome['evidence_frames']),
                              'Qwen window resolution needs evidence inside the faulty window')
                        if outcome['resolution'] == 'UNRESOLVED':
                            check(not any(e['disposition']=='ACCEPT' and set(e['source_episode_ids']) & affected
                                and e['start_frame'] < c['end_frame_exclusive'] and c['start_frame'] < e['end_frame_exclusive']
                                for e in good), 'Unresolved Qwen window cannot be accepted')
                attached = any(i['id']==key for e in good if set(e['source_episode_ids']) & affected for i in e['issues'])
                check((outcome['resolution']=='UNRESOLVED')==attached, 'Unresolved image issue must remain attached')
                if question['kind']=='PHONE_INTERRUPTION':
                    check(all(e['disposition']=='REVIEW' for e in good if set(e['source_episode_ids']) & affected), 'Phone interruption stays REVIEW')
            except (ValueError,KeyError,TypeError) as error:
                bad_parents.update(affected)
                failures.extend(blocked(packet,parents[k],str(error),candidates) for k in affected)
        check(all(o.get('issue_id') in requested for o in outcomes), 'Unknown image issue outcome')
        # A synthetic gap parent can disappear only after images establish non-task context.
        for key, parent in parents.items():
            if (not parent['actions'] and not parent['unlabeled']
                    and all(candidates[r].get('requires_image') and r in excluded for r in parent['evidence_refs'])
                    and all(any(o.get('issue_id')==i['id'] and o.get('resolution')=='RESOLVED'
                                for o in outcomes) for i in parent['issues'])):
                mapped.add(key)
        bad_parents.update(set(parents)-mapped)
        # A split/merge is atomic for its named parents, but unrelated groups continue.
        changed = True
        while changed:
            before = set(bad_parents)
            for e in good:
                if set(e['source_episode_ids']) & bad_parents: bad_parents.update(e['source_episode_ids'])
            changed = before != bad_parents
        good = [e for e in good if not(set(e['source_episode_ids']) & bad_parents)]
        reasons='; '.join(dict.fromkeys(f['reason'] for f in failures)) or 'Image review must cover affected episode identities'
        failures = [blocked(packet,parents[k],reasons,candidates) for k in sorted(bad_parents)]
    else:
        missing = set(candidates)-used-excluded
        for ref in sorted(missing):
            c = candidates[ref]; anchors = c['window_frames']
            failures.append({'episode_id':None,'start_frame':min(anchors),'end_frame_exclusive':max(anchors)+1,
                             'reason':'Every input candidate must be accounted for: '+ref,'source_refs':[ref]})
        # Unknown/unassigned material also withholds overlapping episodes.
        for e in good:
            if any(f['episode_id']!=e['id'] and f['start_frame']<e['end_frame_exclusive']
                   and e['start_frame']<f['end_frame_exclusive'] for f in failures):
                failures.append(blocked(packet,e,'Episode intersects quarantined source interval',candidates))
    bad_ids = {f['episode_id'] for f in failures if f['episode_id'] is not None}
    good = [e for e in good if e['id'] not in bad_ids]
    unique = {digest(f):f for f in failures}
    good.sort(key=lambda e:(e['start_frame'],e['end_frame_exclusive'],e['id']))
    return {'episodes':good,'execution_issues':list(unique.values()),'excluded':result['excluded'],
            'window_adjustments':window_changes,
            'validation':{'status':'PARTIAL' if failures else 'VALID','episodes':len(good),
                          'accepted':sum(e['disposition']=='ACCEPT' for e in good),
                          'issues':sum(len(e['issues']) for e in good),'review_type':result['review_type']}}


def validate(packet, result, *, image_packet=None):
    value = assess(packet,result,image_packet=image_packet)
    check(not value['execution_issues'], '; '.join(i['reason'] for i in value['execution_issues']))
    return value['validation']


def constrain_requests(packet, result):
    """Only mechanical image hints are adjusted; original model bytes stay untouched."""
    result, changes = deepcopy(result), []
    for episode in result.get('episodes',[]):
        for issue in episode.get('issues',[]):
            try:
                a,b = interval(packet,issue)
                allowed = sorted(int(f) for f in packet['sample_frames'] if a<=int(f)<b)
                hints = issue['requested_frames']
                check(isinstance(hints,list), 'Expected frame hint list')
                selected = sorted({f for f in hints if type(f) is int and f in allowed})
                if not selected and allowed: selected=sorted({allowed[0],allowed[len(allowed)//2],allowed[-1]})
                if selected != hints:
                    changes.append({'episode_id':episode['id'],'issue_id':issue['id'],
                                    'field':'requested_frames','before':hints,'after':selected,
                                    'reason':'Real supplied samples inside the half-open issue interval'})
                    issue['requested_frames']=selected
            except (ValueError,KeyError,TypeError):
                pass  # The episode validator records malformed semantic bounds locally.
    return result, changes


def prepare_images(packet_path, result_path, output, *, method_dir=None, issue_ids=None):
    packet, result = read(packet_path), read(result_path)
    assessment = assess(packet,result)
    issues = [i for e in assessment['episodes'] for i in e['issues']]
    if issue_ids is not None:
        issues = [i for i in issues if i['id'] in issue_ids]
        check({i['id'] for i in issues} == set(issue_ids), 'Unknown targeted issue ID')
    check(issues, 'No image issues: skip image review')
    selected = {f for i in issues for f in i['requested_frames']}
    check(selected, 'Image issues must request actual evidence')
    samples = {s['index']: s for s in read(Path(packet['prepared']) / 'samples.json')}
    images = []
    for frame in sorted(selected):
        check(frame in samples, 'Requested frame was not a supplied sample')
        sample = samples[frame]
        file = pin(Path(packet['prepared']) / sample['path'])
        check(file['sha256'] == sample['sha256'], 'Cached image differs from model input')
        images.append({'frame': frame, 'relative_s': sample['relative_s'],
                       'camera_topic': packet['camera_topic'], **file})
    selected_ids = {i['id'] for i in issues}
    affected = [deepcopy(e) for e in assessment['episodes']
                if any(i['id'] in selected_ids for i in e['issues'])]
    method_dir = Path(method_dir).resolve() if method_dir else ROOT / 'skills'
    value = {'kind': 'garment-targeted-image-input-v2', 'text_packet_id': packet['packet_id'],
             'binding': packet['binding'], 'policy': packet['policy'],
             'input_files': [pin(packet_path), pin(result_path), *[{'path': i['path'], 'sha256': i['sha256']} for i in images],
                             *[pin(method_dir / n) for n in METHOD_FILES]],
             'issues': issues, 'episodes': affected, 'images': images,
             'invocation': 'unresolved_text_issues'}
    value['packet_id'] = digest(value)
    write_new(output, value)
    return {'issues': len(issues), 'images': len(images), 'invocation': value['invocation']}


def finalize(packet_path, result_path, output, *, image_packet_path=None, image_result_path=None, image_error=None):
    packet, result = read(packet_path), read(result_path)
    text = assess(packet,result)
    combined = {'packet_id':packet['packet_id'],'review_type':'text',
                'episodes':deepcopy(text['episodes']),'excluded':deepcopy(text['excluded'])}
    failures, image_validation, lineage, evidence = list(text['execution_issues']), None, {}, {}
    ignored_exclusions=[]
    resolved_windows=set()
    if image_packet_path:
        ip = read(image_packet_path); verify_packet(ip)
        check(ip['text_packet_id']==packet['packet_id'], 'Image/text identity mismatch')
        check(pin(result_path) in ip['input_files'], 'Image review was based on a different text result')
        affected = {e['id']:e for e in ip['episodes']}
        ir=None
        if image_error:
            images = {'episodes':[],'execution_issues':[blocked(packet,e,image_error,{c['id']:c for c in packet['candidates']}) for e in affected.values()], 'validation':{'status':'FAILED'}}
        else:
            try:
                ir = read(image_result_path); images = assess(packet,ir,image_packet=ip)
            except (ValueError,KeyError,TypeError) as error:
                images = {'episodes':[],'execution_issues':[blocked(packet,e,str(error),{c['id']:c for c in packet['candidates']}) for e in affected.values()], 'validation':{'status':'FAILED'}}
        image_validation = images['validation']; failures.extend(images['execution_issues'])
        if ir is not None and images['validation']['status'] != 'FAILED':
            bad = {f['episode_id'] for f in images['execution_issues']}
            for ref in {r for q in ip['issues'] for r in q['source_refs']}:
                questions = [q for q in ip['issues'] if ref in q['source_refs']]
                if all(not {e['id'] for e in ip['episodes'] if any(i['id']==q['id'] for i in e['issues'])} & bad
                       and any(o['issue_id']==q['id'] and o['resolution']=='RESOLVED' for o in ir['issue_results'])
                       for q in questions):
                    resolved_windows.add(ref)
        # Every affected parent is replaced by validated children or explicitly quarantined.
        combined['episodes']=[e for e in combined['episodes'] if e['id'] not in affected]
        for original in images['episodes']:
            e=deepcopy(original);lineage[e['id']]=e.pop('source_episode_ids');evidence[e['id']]=ip
            combined['episodes'].append(e)
        updated = {r for e in images['episodes'] for s in e['actions']+e['unlabeled'] for r in s['source_refs']}
        exclusions={e['source_ref']:e for e in combined['excluded'] if e['source_ref'] not in updated}
        if ir is not None:
            candidates={c['id']:c for c in packet['candidates']}
            allowed=referenced(ip['episodes'],candidates) | referenced(ip['issues'],candidates)
            for item in ir['excluded']:
                if item['source_ref'] in allowed:exclusions[item['source_ref']]=item
                else:ignored_exclusions.append({'source_ref':item['source_ref'], 'reason':'Outside affected episode scope; original text decision retained'})
        combined['excluded']=list(exclusions.values())
    reserved={r for f in failures for r in f['source_refs']}
    merged=assess(packet,combined,image_evidence=evidence,reserved_refs=reserved, resolved_windows=resolved_windows)
    failures.extend(merged['execution_issues'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    finalized=[]
    for original in merged['episodes']:
        e=deepcopy(original);model_id=e['id'];parents=lineage.get(model_id,[model_id])
        e['source_episode_ids']=parents
        e['id']='episode-'+digest({'source':packet['binding'],'start':e['start_frame'],'end':e['end_frame_exclusive']})[:16]
        e['start_s']=clock(packet,e['start_frame'])/1e9;e['end_s']=clock(packet,e['end_frame_exclusive'])/1e9
        e['evidence_basis']='targeted_image_review' if model_id in evidence else 'text_from_qwen'
        for action in e['actions']+e['unlabeled']:
            action['start_s']=clock(packet,action['start_frame'])/1e9
            action['end_s']=clock(packet,action['end_frame_exclusive'])/1e9
        finalized.append(e)
    accepted=[e for e in finalized if e['disposition']=='ACCEPT']
    deferred=[e for e in finalized if e['disposition']!='ACCEPT']
    grouped={}
    for failure in failures:
        key=failure['episode_id'] or digest(failure['source_refs'])
        if key in grouped:grouped[key]['reason']='; '.join(dict.fromkeys([grouped[key]['reason'],failure['reason']]))
        else:grouped[key]=deepcopy(failure)
    failures=list(grouped.values())
    binding={'input_packet_id':packet['packet_id'],'source':packet['binding'],'text_result':pin(result_path)}
    for name in ('text-adjustments','image-adjustments','reused-evidence'):
        proof=Path(result_path).parent/(name+'.json')
        if proof.is_file():binding[name]=pin(proof)
    if image_packet_path:
        binding['image_packet']=pin(image_packet_path)
        if image_result_path and Path(image_result_path).is_file():binding['image_result']=pin(image_result_path)
    write_new(out/'episodes.json',{'binding':binding,'episodes':accepted})
    write_new(out/'deferred.json',{'binding':binding,'episodes':deferred,'execution_issues':failures})
    write_new(out/'exclusions.json',merged['excluded'])
    receipt={'status':'LOCAL_REVIEW_COMPLETE','at':time.time(),'binding':binding,'validator':pin(Path(__file__)),
             'text_validation':text['validation'],'image_validation':image_validation,'ignored_image_exclusions':ignored_exclusions,
             'accepted_episodes':len(accepted),'deferred_episodes':len(deferred)+len(failures),
             'actions':sum(len(e['actions']) for e in accepted),'unlabeled_spans':sum(len(e['unlabeled']) for e in accepted),
             'policy_version':packet['policy']['version'],'qwen_requests':0,'hand_requests':0,'previews_generated':0,
             'scope':'Semantic findings plus episode-isolated structural validation; not visual gold or downstream acceptance.'}
    write_new(out/'receipt.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--job-dir', required=True); p.add_argument('--output', required=True)
    for command in ('validate', 'finalize'):
        p = sub.add_parser(command); p.add_argument('--packet', required=True); p.add_argument('--result', required=True)
        p.add_argument('--image-packet'); p.add_argument('--image-result')
        if command == 'finalize': p.add_argument('--output', required=True)
    p = sub.add_parser('images'); p.add_argument('--packet', required=True); p.add_argument('--result', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': value = prepare(args.job_dir, args.output)
    elif args.command == 'images': value = prepare_images(args.packet, args.result, args.output)
    elif args.command == 'finalize': value = finalize(args.packet, args.result, args.output, image_packet_path=args.image_packet, image_result_path=args.image_result)
    else:
        value = validate(read(args.packet), read(args.result))
        if args.image_packet:
            value['image_validation'] = validate(read(args.packet), read(args.image_result), image_packet=read(args.image_packet))
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
