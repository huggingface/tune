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

from .env import MANAGED_ENV_VARIABLES, ENV_VAR_TCMALLOC_LIBRARY_PATH, ENV_VAR_INTEL_OPENMP_LIBRARY_PATH,\
    check_tcmalloc, check_intel_openmp, set_ld_preload_hook
from .cpu import cpu_count_physical, get_procfs_path, get_instances_with_cpu_binding

SEC_TO_NS_SCALE = 1000000000
