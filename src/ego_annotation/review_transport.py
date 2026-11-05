"""Fixed, tool-free Codex CLI calls for review JSON; never runs pipeline work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .review import ISSUE_KINDS, check, pin, write_new


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def output_schema(stage, packet_id, *, source_episode_ids=None):
    string, integer = {'type': 'string'}, {'type': 'integer'}
    strings, integers = {'type': 'array', 'items': string}, {'type': 'array', 'items': integer}
    action = object_schema({'code': string, 'start_frame': integer, 'end_frame_exclusive': integer,
                            'description': string, 'source_refs': strings, 'evidence_frames': integers,
                            'edit': {'type': 'string', 'enum': ['retain', 'normalize', 'merge', 'split', 'supplement']},
                            'reason': string})
    issue = object_schema({'id': string, 'kind': {'type': 'string', 'enum': sorted(ISSUE_KINDS)}, 'question': string,
                           'start_frame': integer, 'end_frame_exclusive': integer,
                           'source_refs': strings, 'requested_frames': integers})
    episode = object_schema({'id': string, 'start_frame': integer, 'end_frame_exclusive': integer,
                             'disposition': {'type': 'string', 'enum': ['ACCEPT', 'REVIEW', 'INCOMPLETE']},
                             'reason': string, 'boundary_notes': string, 'evidence_refs': strings,
                             'actions': {'type': 'array', 'items': action},
                             'issues': {'type': 'array', 'items': issue}})
    episode['properties']['unlabeled'] = {'type': 'array', 'items': object_schema({
        'start_frame': integer, 'end_frame_exclusive': integer, 'reason': string,
        'source_refs': strings, 'evidence_frames': integers})}
    episode['required'].append('unlabeled')
    properties = {'packet_id': {'type': 'string', 'enum': [packet_id]},
                  'review_type': {'type': 'string', 'enum': [stage]},
                  'episodes': {'type': 'array', 'items': episode},
                  'excluded': {'type': 'array', 'items': object_schema({'source_ref': string, 'reason': string})}}
    if stage == 'image':
        parent = {'type':'string'}
        if source_episode_ids is not None: parent['enum']=source_episode_ids
        episode['properties']['source_episode_ids']={'type':'array','items':parent,'minItems':1}
        episode['required'].append('source_episode_ids')
        properties['issue_results'] = {'type': 'array', 'items': object_schema({
            'issue_id': string, 'resolution': {'type': 'string', 'enum': ['RESOLVED', 'UNRESOLVED']},
            'observation': string, 'evidence_frames': integers})}
    return object_schema(properties)


def proc_identity(pid):
    """Only a matching local process start identity proves a call is still running."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in {'Z', 'X'}:
            return None
        return {'pid': pid, 'start_ticks': fields[19]}
    except FileNotFoundError:
        return None


def command(config, folder, image_paths=()):
    folder = Path(folder).resolve()
    argv = [config.get('codex_binary', 'codex'), 'exec', '--ignore-user-config', '--ephemeral',
            '--sandbox', 'read-only', '--skip-git-repo-check', '--cd', str(folder),
            '--model', config['model'], '-c', 'model_reasoning_effort=' + json.dumps(config['reasoning_effort']),
            '-c', 'approval_policy="never"', '-c', 'web_search="disabled"',
            '--json', '--color', 'never', '--output-schema', str(folder / 'schema.json'),
            '--output-last-message', str(folder / 'response.json')]
    for feature in ('shell_tool', 'multi_agent', 'apps', 'remote_plugin', 'hooks', 'memories', 'shell_snapshot'):
        argv += ['--disable', feature]
    for path in image_paths:
        argv += ['--image', str(Path(path).resolve())]
    return argv + ['-']


def invoke(config, folder, prompt, schema, *, image_paths=()):
    """Keep raw output/events, process identity and exit status for durable recovery."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / 'schema.json', schema)
    (folder / 'prompt.txt').write_text(prompt)
    argv = command(config, folder, image_paths)
    started = time.time()
    with (folder / 'events.jsonl').open('wb') as events, (folder / 'stderr.log').open('wb') as stderr:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=events, stderr=stderr,
                                   start_new_session=True)
        identity = proc_identity(process.pid)
        write_new(folder / 'invocation.json', {'started_at': started, 'process': identity,
                  'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
                  'argv': argv, 'attached_images': [pin(p) for p in image_paths]})
        timed_out = False
        try:
            process.communicate(prompt.encode(), timeout=config.get('call_timeout_seconds', 900))
        except subprocess.TimeoutExpired:
            timed_out = True
            if proc_identity(process.pid) == identity:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if proc_identity(process.pid) == identity:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    receipt = {'started_at': started, 'finished_at': time.time(), 'wall_s': time.time() - started,
               'returncode': process.returncode, 'timed_out': timed_out, 'process': identity,
               'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
               'response': pin(folder / 'response.json') if (folder / 'response.json').is_file() else None}
    events = []
    for line in (folder / 'events.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    receipt['thread_id'] = next((e.get('thread_id') for e in events if e.get('type') == 'thread.started'), None)
    receipt['usage'] = next((e.get('usage') for e in reversed(events) if e.get('type') == 'turn.completed'), None)
    receipt['tools_seen'] = sorted({e.get('item', {}).get('type') for e in events
        if e.get('item', {}).get('type') in {'command_execution', 'file_change', 'mcp_tool_call', 'web_search', 'tool_call'}})
    write_new(folder / 'call-receipt.json', receipt)
    check(process.returncode == 0 and not timed_out and receipt['response'],
          f'Codex call failed (exit={process.returncode}, timeout={timed_out}); see {folder}/stderr.log')
    check(not receipt['tools_seen'], 'Unexpected model tool use; reject this call')
    return json.loads((folder / 'response.json').read_text()), receipt
