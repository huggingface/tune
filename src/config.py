from dataclasses import dataclass
from logging import getLogger

from omegaconf import MISSING
from transformers import __version__ as transformers_version

from backends import BackendConfig


LOGGER = getLogger("benchmark")


@dataclass()
class BenchmarkConfig:
    transformers_version: str = transformers_version
    num_runs: int = MISSING
    warmup_runs: int = MISSING
    model: str = MISSING
    batch_size: int = 1
    sequence_length: int = MISSING
    backend: BackendConfig = MISSING
