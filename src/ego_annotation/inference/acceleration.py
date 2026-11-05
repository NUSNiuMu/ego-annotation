"""Observe real DeltaNet dispatch without changing numerical functions or installing packages."""
from collections import Counter
import inspect
from importlib.metadata import PackageNotFoundError, version
import sys


def versions():
    result = {}
    for name in ('flash-linear-attention', 'fla-core', 'causal-conv1d', 'triton'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class DispatchProbe:
    def __init__(self):
        self.counts = Counter()
        self.targets = {}
        self.reason = None
        self.previous = None
        try:
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            from transformers.models.qwen3_5 import modeling_qwen3_5 as delta
            accelerated = dict(causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=causal_conv1d_update,
                               chunk_gated_delta_rule=chunk_gated_delta_rule,
                               fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule)
            for name, fn in accelerated.items():
                self.targets[inspect.unwrap(fn).__code__] = 'accelerated.' + name
            for name in ('causal_conv1d_fn', 'causal_conv1d_update', 'torch_chunk_gated_delta_rule',
                         'torch_recurrent_gated_delta_rule'):
                self.targets[inspect.unwrap(getattr(delta, name)).__code__] = 'reference.' + name
        except (ImportError, AttributeError) as error:
            self.reason = str(error)

    def _profile(self, frame, event, arg):
        if event == 'call' and frame.f_code in self.targets:
            self.counts[self.targets[frame.f_code]] += 1

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is None and self.reason is None:
            sys.setprofile(self._profile)
        elif self.previous is not None:
            self.reason = 'An existing profiler is active; left unchanged'
        return self

    def __exit__(self, *args):
        if self.previous is None and self.reason is None:
            sys.setprofile(None)

    def report(self):
        accelerated = ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
                       'fused_recurrent_gated_delta_rule')
        return {'packages': versions(), 'actual_function_calls': dict(self.counts),
                'all_four_accelerated_observed': all(self.counts['accelerated.' + n] > 0 for n in accelerated),
                'reference_calls': sum(v for k, v in self.counts.items() if k.startswith('reference.')),
                'unavailable_reason': self.reason,
                'timing_note': 'First real window includes Python dispatch profiling overhead; later windows do not.'}
