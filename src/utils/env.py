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

from os import environ
from pathlib import Path

# Environment variables constants
ENV_VAR_TCMALLOC_LIBRARY_PATH = "TCMALLOC_LIBRARY_PATH"
ENV_VAR_INTEL_OPENMP_LIBRARY_PATH = "INTEL_OPENMP_LIBRARY_PATH"

MANAGED_ENV_VARIABLES = {
    "LD_PRELOAD",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "KMP_BLOCKTIME",
    "OMP_MAX_ACTIVE_LEVELS",
    "OMP_NUM_THREADS",
}


def check_tcmalloc() -> Path:
    """
    Ensure tcmalloc library is correctly detected and found
    """
    if ENV_VAR_TCMALLOC_LIBRARY_PATH not in environ:
        raise ValueError(f"Env var {ENV_VAR_TCMALLOC_LIBRARY_PATH} has to be set to location of libtcmalloc.so")

    if len(environ[ENV_VAR_TCMALLOC_LIBRARY_PATH]) == 0:
        raise ValueError(f"Env var {ENV_VAR_TCMALLOC_LIBRARY_PATH} cannot be empty")

    tcmalloc_path = Path(environ[ENV_VAR_TCMALLOC_LIBRARY_PATH])
    if not tcmalloc_path.exists():
        raise ValueError(
            f"Path {tcmalloc_path.as_posix()} pointed by "
            f"env var {ENV_VAR_TCMALLOC_LIBRARY_PATH} doesn't exist"
        )

    return tcmalloc_path


def check_intel_openmp() -> Path:
    """
    Ensure Intel OpenMP library is correctly detected and found
    """
    if ENV_VAR_INTEL_OPENMP_LIBRARY_PATH not in environ:
        raise ValueError(f"Env var {ENV_VAR_INTEL_OPENMP_LIBRARY_PATH} has to be set to location of libomp.so")

    if len(environ[ENV_VAR_INTEL_OPENMP_LIBRARY_PATH]) == 0:
        raise ValueError(f"Env var {ENV_VAR_INTEL_OPENMP_LIBRARY_PATH} cannot be empty")

    intel_openmp_path = Path(environ[ENV_VAR_INTEL_OPENMP_LIBRARY_PATH])
    if not intel_openmp_path.exists():
        raise ValueError(
            f"Path {intel_openmp_path.as_posix()} pointed by "
            f"env var {ENV_VAR_INTEL_OPENMP_LIBRARY_PATH} doesn't exist"
        )

    return intel_openmp_path


def set_ld_preload_hook(config):
    ld_preload = []
    if hasattr(config, "malloc") and "tcmalloc" == config.malloc.name:
        from utils import check_tcmalloc
        tcmalloc_path = check_tcmalloc()
        ld_preload.append(tcmalloc_path.as_posix())

    if hasattr(config, "openmp_backend") and "intel" == config.openmp_backend.name:
        from utils import check_intel_openmp
        intel_omp_path = check_intel_openmp()
        ld_preload.append(intel_omp_path.as_posix())

    ld_preload_str = " ".join(ld_preload)
    if "LD_PRELOAD" in environ:
        ld_preload_str += " " + environ.get("LD_PRELOAD", default="")

    environ["LD_PRELOAD"] = ld_preload_str