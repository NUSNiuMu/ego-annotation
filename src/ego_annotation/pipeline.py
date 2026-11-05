#!/usr/bin/env python3
"""Batch MCAP -> sampled RGB -> Qwen, with persistent jobs and window checkpoints."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from ego_annotation import sampling as rgb_sample
from ego_annotation.inference.schema import parse_response
from ego_annotation.config import PipelineConfig

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FILES = ('pipeline.py', 'config.py', 'sampling.py', 'qwen_batch.py', 'gpu_devices.py', 'inference/worker.py',
                 'inference/schema.py', 'inference/attention.py', 'inference/acceleration.py', 'inference/model_identity.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    """Atomic checkpoints: a killed process cannot leave a partial JSON receipt."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Already running: {path}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def discover(sources, batch_roots):
    """Each directory containing MCAP parts is one recording; files are single-part jobs."""
    groups = []
    for source in sources:
        source = Path(source).resolve()
        parts = sorted(source.glob('*.mcap')) if source.is_dir() else [source]
        if not parts or any(not p.is_file() or p.suffix != '.mcap' for p in parts):
            raise ValueError(f'No readable MCAP input: {source}')
        groups.append(parts)
    for parent in batch_roots:
        parent = Path(parent).resolve()
        if not parent.is_dir():
            raise ValueError(f'Batch root is not a directory: {parent}')
        folders = sorted({p.parent for p in parent.rglob('*.mcap') if p.is_file()})
        groups.extend(sorted(p.glob('*.mcap')) for p in folders)
    if not groups:
        raise ValueError('No MCAP recordings found')
    # Also avoid scanning an explicitly supplied recording twice in the same batch.
    return list({tuple(p.resolve() for p in group): group for group in groups}.values())


def identify(parts):
    files = []
    for path in parts:
        before = path.stat()
        sha = file_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed while hashing: {path}')
        files.append({'path': str(path.resolve()), 'name': path.name, 'bytes': after.st_size,
                      'mtime_ns': after.st_mtime_ns, 'sha256': sha})
    source_id = digest([{'bytes': f['bytes'], 'sha256': f['sha256']} for f in files])
    return {'source_id': source_id, 'files': files}


def ensure_source_unchanged(source):
    for item in source['files']:
        stat = Path(item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise RuntimeError(f"Source changed during preparation: {item['path']}")


def runtime_fingerprint():
    return {name: file_hash(ROOT / name) for name in RUNTIME_FILES}


def model_fingerprint(model_dir):
    """Separate jobs when model files or the inference environment change."""
    if model_dir is None:
        return None
    from importlib.metadata import PackageNotFoundError, version
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise ValueError(f'Model directory does not exist: {model_dir}')
    files = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            if path.suffix in ('.json', '.jinja', '.txt'):
                files[path.name] = {'sha256': file_hash(path)}
            elif path.suffix == '.safetensors':
                stat = path.stat()
                files[path.name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'triton', 'flash-linear-attention', 'causal-conv1d'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'files': files, 'packages': packages,
            'note': 'Weight sizes/mtime identify normal file changes; weight contents are not hashed.'}


def archive(path):
    if path.exists():
        path.rename(path.with_name(path.name + '.previous-' + str(time.time_ns())))


def prepared_valid(folder, source_id, prep_hash):
    try:
        receipt = read(folder / 'receipt.json')
        if receipt['source_id'] != source_id or receipt['preparation_sha256'] != prep_hash:
            return False
        return all(file_hash(folder / name) == sha for name, sha in receipt['files'].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(parts, folder, source, config, prep_hash):
    if prepared_valid(folder, source['source_id'], prep_hash):
        return read(folder / 'timing.json'), True
    # Only this job's own incomplete/cache directory is replaced. Keep prior attempts.
    archive(folder)
    incomplete = folder.with_name('prepared.incomplete')
    archive(incomplete)
    timing = rgb_sample.run(parts, config.topic, incomplete, config.interval_s,
                            config.longest_edge, config.window_s, config.decode_threads,
                            quality=config.jpeg_quality)
    ensure_source_unchanged(source)
    files = {str(p.relative_to(incomplete)): file_hash(p)
             for p in sorted(incomplete.rglob('*')) if p.is_file()}
    save(incomplete / 'receipt.json', {'source_id': source['source_id'],
                                     'preparation_sha256': prep_hash, 'files': files})
    incomplete.rename(folder)
    return timing, False


def checked_result(path, binding, window):
    """Validate immutable execution evidence, including unsuccessful model output."""
    path = Path(path)
    result = read(path)
    if (result['binding'] != binding or result['source_window'] != window
            or result['window'] != int(path.stem)):
        raise ValueError('Window binding mismatch')
    if result['status'] == 'NO_INPUT':
        if window['indices'] or result['parsed'] != {'segments': [], 'uncertainties': []}:
            raise ValueError('Invalid empty Qwen window')
    else:
        if (not window['indices'] or type(result['truncated']) is not bool
                or not isinstance(result['raw_response'], str)
                or ('input_frames' in result and result['input_frames'] != window['indices'])):
            raise ValueError('Invalid model output record')
        parsed, error, *_ = parse_response(result['raw_response'], window['indices'])
        expected = 'TRUNCATED' if result['truncated'] else ('PASS' if error is None else 'INVALID_JSON_OR_SCHEMA')
        if result['status'] != expected or result['parsed'] != parsed:
            raise ValueError('Raw/parsed model result mismatch')
    if 'reused_from' in result:
        original = path.parent.parent / 'reused' / path.name
        job = read(path.parent.parent / 'job.json')
        reuse = job['execution']['reuse']
        prior = read(path.parent.parent / 'reused' / 'job.json')
        if (file_hash(path.parent.parent / 'reused' / 'job.json') != reuse['job_sha256']
                or prior['binding']['job_id'] != reuse['job_id']
                or file_hash(original) != reuse['windows'][path.stem]
                or result['reused_from'] != {'job_id': reuse['job_id'], 'sha256': file_hash(original)}):
            raise ValueError('Reused window provenance differs')
        old = checked_result(original, prior['binding'], window)
        if result != {**old, 'binding': binding, 'reused_from': result['reused_from']}:
            raise ValueError('Reused model output changed')
    return result


def result_valid(path, binding, window):
    try:
        checked_result(path, binding, window)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def process_job(parts, work_dir, config, code, model_id, get_worker, prepare_only=False, reuse_job=None):
    source = identify(parts)
    config_hash = digest(config.snapshot())
    prep_hash = digest({'source_parameters': {k: config.snapshot()[k] for k in (
        'topic', 'window_s', 'interval_s', 'longest_edge', 'jpeg_quality', 'decode_threads', 'transport')},
        'sampler': code['sampling.py'],
        'environment': {'av': rgb_sample.av.__version__, 'mcap': rgb_sample.mcap.__version__,
                        'pillow': rgb_sample.pillow_version,
                        'libav': {k: list(v) for k, v in rgb_sample.av.library_versions.items()}}})
    # Preparation has its own identity so --prepare-only output is reusable for inference.
    source_dir = work_dir / 'sources' / source['source_id']
    source_dir.mkdir(parents=True, exist_ok=True)
    folder = source_dir / ('prepare-' + prep_hash)
    execution = {'config_sha256': config_hash, 'preparation_sha256': prep_hash,
                 'code': code, 'model': model_id}
    prior = None
    if reuse_job is not None:
        reuse_job = Path(reuse_job).resolve()
        prior = read(reuse_job / 'job.json')
        if (prior['source'] != source or any(prior['execution'][k] != execution[k]
                for k in ('config_sha256', 'preparation_sha256', 'model'))):
            raise ValueError('Reuse source, preparation, configuration or model differs')
        if prior['execution'].get('reuse'):
            raise ValueError('Reuse must name original inference evidence')
        execution['reuse'] = {'job_id': prior['binding']['job_id'], 'job_sha256': file_hash(reuse_job/'job.json'),
                              'windows': {p.stem: file_hash(p) for p in sorted((reuse_job/'windows').glob('*.json'))}}
    job_id = digest({'source_id': source['source_id'], 'execution': execution})
    job = work_dir / 'jobs' / job_id
    job.mkdir(parents=True, exist_ok=True)
    binding = {'job_id': job_id, 'source_id': source['source_id'],
               'config_sha256': config_hash, 'execution_sha256': digest(execution)}
    status = {'job_id': job_id, 'source_id': source['source_id'], 'job_dir': str(job),
              'source': [str(p) for p in parts], 'status': 'PREPARING', 'completed_windows': 0}
    with lock(job / '.lock'), lock(source_dir / '.lock'):
        if (job / 'status.json').is_file() and read(job / 'status.json')['status'] == 'COMPLETE' and prepared_valid(folder, source['source_id'], prep_hash):
            previous = read(job / 'job.json')
            if previous['binding'] != binding or previous['execution'] != execution:
                raise ValueError('Completed job binding changed')
            windows = read(folder / 'windows.json')
            for i, window in enumerate(windows):
                checked_result(job / 'windows' / f'{i:06d}.json', binding, window)
            # Review packets pin these files; reuse must not rewrite their timestamps or bytes.
            return {**read(job / 'status.json'), 'reused_windows': len(windows), 'preparation_reused': True}
        save(job / 'job.json', {'binding': binding, 'config': config.snapshot(),
                               'execution': execution, 'source': source, 'prepared': str(folder)})
        save(job / 'status.json', status)
        try:
            timing, reused = prepare(parts, folder, source, config, prep_hash)
            windows = read(folder / 'windows.json')
            samples = {s['index']: s for s in read(folder / 'samples.json')}
            if prepare_only:
                status.update(status='PREPARED', windows=len(windows), sampled_images=len(samples),
                              preparation_reused=reused)
                save(job / 'status.json', status)
                return status
            results_dir = job / 'windows'
            results_dir.mkdir(exist_ok=True)
            if prior is not None:
                originals = job / 'reused'
                originals.mkdir(exist_ok=True)
                for name in ('job.json', *[f'windows/{i}.json' for i in execution['reuse']['windows']]):
                    src = reuse_job / name
                    dest = originals / src.name
                    expected_hash = (execution['reuse']['job_sha256'] if name == 'job.json'
                                     else execution['reuse']['windows'][src.stem])
                    if file_hash(src) != expected_hash or (dest.exists() and file_hash(dest) != expected_hash):
                        raise ValueError('Reuse evidence changed')
                    if not dest.exists(): shutil.copyfile(src, dest)
                    if name == 'job.json': continue
                    i = int(src.stem)
                    old = checked_result(src, prior['binding'], windows[i])
                    dest = results_dir / src.name
                    value = {**old, 'binding': binding,
                             'reused_from': {'job_id': prior['binding']['job_id'], 'sha256': expected_hash}}
                    if dest.exists():
                        if read(dest) != value: raise ValueError('Existing reuse differs')
                    else: save(dest, value)
            pending = [i for i, w in enumerate(windows)
                       if not result_valid(results_dir / f'{i:06d}.json', binding, w)]
            failed = [i for i in pending if (results_dir / f'{i:06d}.json').exists()]
            if failed:
                raise RuntimeError(f'Untrusted saved window records {failed}; no inference was repeated')
            results = []
            for i, window in enumerate(windows):
                path = results_dir / f'{i:06d}.json'
                if result_valid(path, binding, window):
                    results.append(read(path))
                    continue
                status.update(status='INFERENCING', current_window=i, completed_windows=len(results), windows=len(windows))
                save(job / 'status.json', status)
                if not window['indices']:
                    result = {'status': 'NO_INPUT', 'source_window': window,
                              'parsed': {'segments': [], 'uncertainties': []}}
                else:
                    worker = get_worker()
                    save(job / 'runtime.json', worker.runtime)
                    started = time.perf_counter()
                    images = [rgb_sample.consume_jpeg_path(folder / samples[k]['path']) for k in window['indices']]
                    read_s = time.perf_counter() - started
                    result = worker.infer(window, samples, images)
                    save(job / 'runtime.json', worker.runtime)
                    result['runtime'] = worker.runtime
                    result['jpeg_read_decode_s'] = read_s
                result.update(binding=binding, config=config.snapshot(), model_revision_baseline=config.model_revision,
                              window=i, source={'source_id': source['source_id'],
                                                'frames': [samples[k] for k in window['indices']]})
                archive(path)
                save(path, result)
                checked_result(path, binding, window)
                results.append(result)
                print(json.dumps({'job_id': job_id, 'window': i, 'status': result['status']}, ensure_ascii=False), flush=True)
            summary = {'status': 'COMPLETE', 'binding': binding, 'windows': len(results),
                       'action_segments': sum(len(r['parsed']['segments']) for r in results if r['status']=='PASS'),
                       'review_windows': [r['window'] for r in results if r['status'] not in ('PASS','NO_INPUT')],
                       'source_duration_s': timing['input_duration_s'], 'preparation': timing,
                       'preparation_reused': reused,
                       'processor_to_output_wall_s': sum(r.get('processor_to_output_wall_s', 0) for r in results),
                       'generate_prefill_decode_s': sum(r.get('generate_prefill_decode_s', 0) for r in results),
                       'jpeg_read_decode_s': sum(r.get('jpeg_read_decode_s', 0) for r in results),
                       'output_tokens': sum(r.get('output_tokens', 0) for r in results),
                       'quality_scope': 'All windows accounted for; invalid outputs require targeted review. No episode success claim.'}
            save(job / 'summary.json', summary)
            save(job / 'segments.json', [{'window': r['window'], 'source_window': r['source_window'],
                                         **(r['parsed'] if r['status'] in ('PASS','NO_INPUT') else
                                            {'segments': [], 'uncertainties': [], 'review_required': r['status']})}
                                        for r in results])
            status.update(status='COMPLETE', completed_windows=len(results), windows=len(windows),
                          review_windows=summary['review_windows'],
                          reused_windows=len(windows) - len(pending), preparation_reused=reused)
            save(job / 'status.json', status)
            return status
        except BaseException as error:
            status.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'FAILED',
                          error=str(error))
            save(job / 'status.json', status)
            raise


def run_batch(groups, work_dir, config, get_worker, model_id=None, prepare_only=False, reuse_job=None):
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    code = runtime_fingerprint()
    results = []
    seen = set()
    for parts in groups:
        try:
            # process_job also checks the content identity, so a moved/copied recording deduplicates.
            result = process_job(parts, work_dir, config, code, model_id, get_worker, prepare_only, reuse_job)
        except Exception as error:
            result = {'source': [str(p) for p in parts], 'status': 'FAILED', 'error': str(error)}
        if result.get('job_id') in seen:
            result = {**result, 'duplicate_in_batch': True}
        if result.get('job_id'):
            seen.add(result['job_id'])
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        save(work_dir / 'batch.json', {'config': config.snapshot(), 'recordings': results,
                                     'failed': sum(r['status'] == 'FAILED' for r in results)})
    return results
