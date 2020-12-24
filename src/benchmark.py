from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from time import time_ns
from typing import List

from pandas import Series, to_timedelta


LOGGER = getLogger("benchmark")


@dataclass
class Benchmark:
    results: List[int] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.results)

    @contextmanager
    def track(self):
        start = time_ns()
        yield
        end = time_ns()

        # Append the time to the buffer
        self.results.append(end - start)

        LOGGER.info(f"Tracked function took: {(end - start)}ns")

    def to_pandas(self) -> Series:
        return to_timedelta(Series(self.results, name="inference_time (ns)"), "ns")
