#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from time import perf_counter_ns
from typing import List

from pandas import DataFrame

from utils import SEC_TO_NS_SCALE

LOGGER = getLogger("benchmark")


@dataclass
class Benchmark:
    latencies: List[float] = field(default_factory=list)
    throughput: float = float("inf")

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self):
        start = perf_counter_ns()
        yield
        end = perf_counter_ns()

        # Append the time to the buffer
        self.latencies.append(end - start)

        LOGGER.debug(f"Tracked function took: {(end - start)}ns")

    def finalize(self, duration_ns: int):
        self.throughput = round((len(self.latencies) / duration_ns) * SEC_TO_NS_SCALE, 2)

    def to_pandas(self) -> DataFrame:
        return DataFrame({
            "latency": self.latencies,
            "throughput": [self.throughput] * len(self.latencies)
        })
