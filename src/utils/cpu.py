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
from itertools import chain
from typing import List, Tuple


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
    cores_per_socket = {}
    current_info = {}
    with open(f'{get_procfs_path()}/cpuinfo', "rb") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                # print(current_info)
                # new section
                if b'physical id' in current_info and b'cpu cores' in current_info:
                    cores_per_socket[current_info[b'physical id']] = current_info[b'cpu cores']

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

    total_num_cores = sum(cores_per_socket.values())
    core_to_socket_mapping = {}
    for physical, logical in physical_logical_mapping.items():
        physical_id = physical // 1000

        if physical_id not in core_to_socket_mapping:
            core_to_socket_mapping[physical_id] = set()

        core_to_socket_mapping[physical_id].add(logical)

    return total_num_cores, cores_per_socket, core_to_socket_mapping


def get_instances_with_cpu_binding(num_core_per_instance: int = -1, num_instances: int = 1) -> List[Tuple[List[int], List[int]]]:
    """
    :param num_core_per_instance: Number of cores to use per instances, -1 means "use all the CPU cores"
    :param num_instances: Number of model instances to distribute CPU cores for
    :return: List[List[int]] Per instance list of CPU core affinity
    """
    total_num_cores, cores_per_socket, core_to_socket_mapping = cpu_count_physical()
    instance_binding = []

    # items in a set are unique, if their more than 1 value, then we have different number core per socket.
    assert len(set(cores_per_socket.values())) == 1, "CPU cores are not equal across sockets"

    # No special information given to restrict number of core -> Use all the cores
    if num_core_per_instance < 0:
        # We set the number of core per instance to the number of core of one single socket.
        num_core_per_instance = cores_per_socket[0]
        need_multiple_socket_per_instance = False
        need_socket_overcommit = num_instances > 1  # Asking for more than one instance with all the cores

    # Number of core span more than a single socket
    elif num_core_per_instance > cores_per_socket[0]:
        num_core_per_instance = max(num_core_per_instance, total_num_cores)
        need_multiple_socket_per_instance = len(cores_per_socket) > 1  # Ensure we have multiple socket
        need_socket_overcommit = num_instances > 1

    # Span over only on socket
    else:
        need_multiple_socket_per_instance = False
        need_socket_overcommit = num_core_per_instance * num_instances > cores_per_socket[0]

    for instance in range(num_instances):
        # On which socket to allocate the instance
        if need_multiple_socket_per_instance:
            socket = list(core_to_socket_mapping.keys())
            cores = {c for s in socket for c in core_to_socket_mapping[s]}

        else:
            # {socket_id -> [cores]}
            socket = [instance % len(cores_per_socket.keys())]

            # Get the list of available cores (unallocated) on the target socket
            cores = core_to_socket_mapping[socket[0]]

        # Pop out allocated core
        # Overcommiting does pop out cores because it will have overlap between instances
        # Overcommiting doesnt attempt to do things smart limiting the overhead.
        if need_socket_overcommit:
            cores_it = iter(cores)
            bindings = [next(cores_it) for i in range(num_core_per_instance)]
        else:
            bindings = [cores.pop() for _ in range(num_core_per_instance)]

        instance_binding.append((socket, bindings))

    return instance_binding