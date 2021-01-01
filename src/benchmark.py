from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from time import perf_counter
from typing import List

from pandas import Series


LOGGER = getLogger("benchmark")


@dataclass
class Benchmark:
    results: List[float] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.results)

    @contextmanager
    def track(self):
        start = perf_counter()
        yield
        end = perf_counter()

        # Append the time to the buffer
        self.results.append(end - start)

        LOGGER.debug(f"Tracked function took: {(end - start)}ns")

    def to_pandas(self) -> Series:
        return Series(self.results, name="inference_time_secs")
