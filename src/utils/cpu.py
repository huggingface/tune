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
import numpy as np
import platform
import re
import subprocess
import sys
from logging import getLogger
from os import getpid
from typing import List, Tuple

LOGGER = getLogger("cpu")


class CPUinfo:
    def __init__(self):
        self.cpuinfo = []

        if platform.system() == "Windows":
            raise RuntimeError("Windows platform is not supported!!!")
        elif platform.system() == "Linux":
            args = ["lscpu", "--parse=CPU,Core,Socket,Node"]
            lscpu_info = subprocess.check_output(args, universal_newlines=True).split("\n")

            # Get information about  cpu, core, socket and node
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]+)"
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))

            self._get_socket_info()

    def _get_socket_info(self):

        self.socket_physical_cores = []  # socket_id is index
        self.socket_logical_cores = []   # socket_id is index
        self.sockets = int(max([line[2] for line in self.cpuinfo])) + 1
        self.core_to_sockets = {}

        for socket_id in range(self.sockets):
            cur_socket_physical_core = []
            cur_socket_logical_core = []

            for line in self.cpuinfo:
                if socket_id == int(line[2]):
                    if line[1] not in cur_socket_physical_core:
                        cur_socket_physical_core.append(line[1])

                    cur_socket_logical_core.append(line[0])

                self.core_to_sockets[line[0]] = line[2]

            self.socket_physical_cores.append(cur_socket_physical_core)
            self.socket_logical_cores.append(cur_socket_logical_core)

    @property
    def socket_nums(self):
        return self.sockets

    @property
    def physical_core_nums(self):
        return len(self.socket_physical_cores) * len(self.socket_physical_cores[0])

    @property
    def logical_core_nums(self):
        return len(self.socket_logical_cores) * len(self.socket_logical_cores[0])

    @property
    def get_all_physical_cores(self):
        return np.array(self.socket_physical_cores).flatten().tolist()

    @property
    def get_all_logical_cores(self):
        return np.array(self.socket_logical_cores).flatten().tolist()

    def get_socket_physical_cores(self, socket_id):
        if socket_id < 0 or socket_id > self.sockets - 1:
            LOGGER.error(f"Invalid socket id {socket_id}")
        return self.socket_physical_cores[socket_id]

    def get_socket_logical_cores(self, socket_id):
        if socket_id < 0 or socket_id > self.sockets - 1:
            LOGGER.error(f"Invalid socket id {socket_id}")
        return self.socket_logical_cores[socket_id]

    def get_sockets_for_cores(self, core_ids):
        return {self.core_to_sockets[core] for core in core_ids}


def get_procfs_path():
    """Return updated psutil.PROCFS_PATH constant."""
    """Copied from psutil code, and modified to fix an error."""
    return sys.modules['psutil'].PROCFS_PATH


def cpu_count_physical():
    """Return the number of physical cores in the system."""
    """Copied from psutil code, and modified to fix an error."""

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

    # Matching ICX
    total_num_cores = 64
    cores_per_socket = {0: 32, 1: 32}
    core_to_socket_mapping = {0: set(range(32)), 1: set(range(32, 64))}

    # 64
    # {0: 32, 1: 32}
    # {
    #   0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    #   1: {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}
    # }

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
        need_socket_overcommit = num_core_per_instance > cores_per_socket[0]

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


def configure_numa(socket_binding: List[int], core_binding: List[int]):
    from numa import available as is_numa_available, set_membind, get_membind, set_affinity, get_affinity
    if is_numa_available():
        LOGGER.info("Configuring NUMA:")

        pid = getpid()

        # Set core binding affinity
        set_affinity(pid, set(core_binding))
        LOGGER.info(f"\tScheduler affinity set to: {get_affinity(pid)}")

        # Set memory allocation affinity
        set_membind(set(socket_binding))
        LOGGER.info(f"\tBinding memory allocation on {get_membind()}")
    else:
        LOGGER.info("NUMA not available on the system, skipping configuration")

    # Configure taskset
    # TODO: Check with @Sangeeta if this is still needed as we set CPU scheduler affinity above
    # system(f"taskset -p -c {','.join(map(str, core_binding))} {getpid()}")
    # LOGGER.info(f"[TASKSET] Set CPU affinity to: {core_binding} (pid={getpid()})")