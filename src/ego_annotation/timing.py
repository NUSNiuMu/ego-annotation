"""Append operational spans; timing is never downstream acceptance evidence."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys
import time
import uuid

_parent = ContextVar('timing_parent', default=None)


def emit(path, event):
    if path is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(event, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except OSError as error:
        # A full/unavailable timing destination must not invalidate real work.
        print(json.dumps({'timing_error': type(error).__name__, 'errno': error.errno}),
              file=sys.stderr, flush=True)


@contextmanager
def span(path, stage, **fields):
    details = {}
    if path is None:
        yield details
        return
    started, tick = time.time(), time.perf_counter()
    common = {'span_id': uuid.uuid4().hex, 'stage': stage, 'pid': os.getpid(),
              'parent_span_id': _parent.get(), 'started_at': started, **fields}
    token = _parent.set(common['span_id'])
    emit(path, {**common, 'status': 'STARTED'})
    status, error_type = 'COMPLETE', None
    try:
        yield details
    except BaseException as error:
        status, error_type = 'FAILED', type(error).__name__
        raise
    finally:
        emit(path, {**common, **details, 'status': status, 'error_type': error_type,
                    'finished_at': time.time(), 'wall_s': time.perf_counter()-tick})
        _parent.reset(token)
