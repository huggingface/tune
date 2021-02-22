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
import logging

from multiprocess.connection import Pipe
from multiprocess.context import Process
from pathlib import Path
from typing import Sequence, Optional, Any, List

from hydra import TaskFunction
from hydra.core.config_loader import ConfigLoader
from hydra.core.singleton import Singleton

from hydra.core.utils import JobReturn, setup_globals, configure_log
from hydra.plugins.launcher import Launcher
from omegaconf import DictConfig

from ._core import execute_job


log = logging.getLogger(__name__)


class ProcessLauncher(Launcher):
    def __init__(self, **kwargs: Any) -> None:
        """Process Launcher

        Launches parallel jobs using Joblib.Parallel. For details, refer to:
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html

        This plugin is based on the idea and inital implementation of @emilemathieutmp:
        https://github.com/facebookresearch/hydra/issues/357
        """
        self.config: Optional[DictConfig] = None
        self.config_loader: Optional[ConfigLoader] = None
        self.task_function: Optional[TaskFunction] = None

    def setup(self, config: DictConfig, config_loader: ConfigLoader, task_function: TaskFunction) -> None:
        self.config = config
        self.config_loader = config_loader
        self.task_function = task_function

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
        setup_globals()

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Launching { len(job_overrides)} jobs")
        log.info("Launching jobs, sweep output dir : {}".format(sweep_dir))

        singleton_state = Singleton.get_state()

        runs, (pipe_reader, pipe_writer) = [], Pipe(duplex=False)
        for idx, overrides in enumerate(job_overrides):
            job_kwargs = {
                "idx": initial_job_idx + idx,
                "overrides": overrides,
                "config_loader": self.config_loader,
                "config": self.config,
                "task_function": self.task_function,
                "singleton_state": singleton_state,
                "pipe": pipe_writer
            }

            p = Process(target=execute_job, kwargs=job_kwargs)
            try:

                p.start()
                job_result = pipe_reader.recv()

                # Retrieve from pipe
                if isinstance(job_result, JobReturn):
                    runs.append(job_result)
                else:
                    log.warning(f"Error while running benchmark [{idx}]: {overrides} -> {job_result}")
            finally:
                p.join()

        assert isinstance(runs, List)
        for run in runs:
            assert isinstance(run, JobReturn)
        return runs