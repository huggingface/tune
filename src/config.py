from dataclasses import dataclass
from logging import getLogger

from omegaconf import MISSING

from backends import BackendConfig


LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig:
    model: str = MISSING
    sequence_length: int = MISSING
    backend: BackendConfig = MISSING
    num_runs: int = 10
    warmup_runs: int = 5
