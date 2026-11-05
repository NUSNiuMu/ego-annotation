"""Confirmed workflow parameters; experiments have separate entry points."""
from dataclasses import asdict, dataclass, field


SYSTEM_PROMPT = (
    'You are a strict JSON serializer. Output exactly one JSON object. '
    'The first character must be { and the last character must be }. '
    'Never output Markdown, code fences, backticks, explanations, or reasoning.'
)
ASSISTANT_PREFIX = '{"segments":'


@dataclass(frozen=True)
class PipelineConfig:
    topic: str = '/egohalo/record/cam0/h264'
    window_s: float = 3.0
    interval_s: float = 0.5
    longest_edge: int = 960
    jpeg_quality: int = 80
    decode_threads: int = 1
    cpu_threads: int = 4
    transport: str = field(default='jpeg-disk', init=False)
    model_name: str = 'Qwen3.5-9B'
    model_revision: str = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    dtype: str = 'float16'
    attention: str = 'sdpa'
    enable_thinking: bool = False
    do_sample: bool = False
    use_cache: bool = True
    max_new_tokens: int = 2048
    processor_min_pixels: int = 65536
    processor_max_pixels: int = 522240

    def snapshot(self):
        return asdict(self)
