from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Generic, TypeVar, ClassVar

from omegaconf import MISSING
from psutil import cpu_count

from benchmark import Benchmark

LOGGER = getLogger("backends")


@dataclass
class BackendConfig(ABC):
    name: str = MISSING
    device: str = MISSING
    precision: str = MISSING

    num_threads: int = cpu_count()
    num_interops_threads: int = cpu_count()


class Backend(ABC):
    NAME: ClassVar[str]

    @abstractmethod
    def execute(self, config: BackendConfig) -> Benchmark:
        raise NotImplementedError()
