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

# Copied from FastFormer code: https://github.com/microsoft/fastformers/blob/main/examples/fastformers/run_superglue.py
import sys
from typing import List


def get_procfs_path():
    """Return updated psutil.PROCFS_PATH constant."""
    """Copied from psutil code, and modified to fix an error."""
    return sys.modules['psutil'].PROCFS_PATH


def cpu_count_physical():
    """Return the number of physical cores in the system."""
    """Copied from psutil code, and modified to fix an error."""
    # Method #1 doesn't work for some dual socket topologies.
    # # Method #1
    # core_ids = set()
    # for path in glob.glob(
    #         "/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
    #     with open_binary(path) as f:
    #         core_ids.add(int(f.read()))
    # result = len(core_ids)
    # if result != 0:
    #     return result

    # Method #2
    physical_logical_mapping = {}
    mapping = {}
    current_info = {}
    with open(f'{get_procfs_path()}/cpuinfo', "rb") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                # print(current_info)
                # new section
                if b'physical id' in current_info and b'cpu cores' in current_info:
                    mapping[current_info[b'physical id']] = current_info[b'cpu cores']

                if b'physical id' in current_info and b'core id' in current_info and b'processor' in current_info:
                    # print(current_info[b'physical id'] * 1000 + current_info[b'core id'])
                    if current_info[b'physical id'] * 1000 + current_info[b'core id'] not in physical_logical_mapping:
                        physical_logical_mapping[
                            current_info[b'physical id'] * 1000 + current_info[b'core id']
                        ] = current_info[b'processor']
                current_info = {}
            else:
                # ongoing section
                if (line.startswith(b'physical id') or
                        line.startswith(b'cpu cores') or
                        line.startswith(b'core id') or
                        line.startswith(b'processor')):
                    key, value = line.split(b'\t:', 1)
                    current_info[key.rstrip()] = int(value.rstrip())

    physical_processor_ids = []
    for key in sorted(physical_logical_mapping.keys()):
        physical_processor_ids.append(physical_logical_mapping[key])

    result = sum(mapping.values())
    # return result or None  # mimic os.cpu_count()
    return result, physical_processor_ids


def get_instances_with_cpu_binding(num_threads: int = -1, num_instances: int = 1) -> List[List[int]]:
    """
    :param num_threads: Number of threads to use per instances, -1 means "use all the CPU cores"
    :param num_instances: Number of model instances to distribute CPU cores for
    :return: List[List[int]] Per instance list of CPU core affinity
    """
    num_cores, processor_list = cpu_count_physical()

    # Use all the cores
    if num_threads < 0:
        num_threads = num_cores

    return [
        [(instance * num_threads) + i for i in range(num_threads)]
        for instance in range(num_instances)
    ]