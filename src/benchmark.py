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
