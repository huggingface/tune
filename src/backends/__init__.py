from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Generic, TypeVar, ClassVar

from hydra.types import TargetConf
from omegaconf import MISSING
from psutil import cpu_count

from benchmark import Benchmark

LOGGER = getLogger("backends")


@dataclass
class BackendConfig(TargetConf):
    name: str = MISSING
    device: str = MISSING
    precision: str = MISSING

    num_threads: int = cpu_count()
    num_interops_threads: int = cpu_count()


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    @abstractmethod
    def configure(self, config: BackendConfigT):
        raise NotImplementedError()

    @abstractmethod
    def execute(self, config: BackendConfig) -> Benchmark:
        raise NotImplementedError()

    @classmethod
    def allocate(cls, config: 'BenchmarkConfig'):
        pass
