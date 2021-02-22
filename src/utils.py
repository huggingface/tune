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
from pathlib import Path

SEC_TO_NS_SCALE = 1000000000

# Environment variables constants
ENV_VAR_TCMALLOC_LIBRARY_PATH = "TCMALLOC_LIBRARY_PATH"


def check_tcmalloc():
    from os import environ
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
