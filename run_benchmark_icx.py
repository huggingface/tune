#  Copyright 2021 Intel Corporation.
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

import subprocess
from argparse import ArgumentParser
import find_best_setup

#Predefined backend lists; Please change as needed

BACKEND_SET_CHOICES = {"pt", "tf", "pcl", "bert_cpp"}
pt_experiment_backends = ['pytorch', 'torchscript']
tf_experiment_backends = ['tensorflow', 'xla']
pcl_experiment_backends = ['pcl', 'pytorch', 'torchscript']
bert_cpp_experiment_backends = ['fused', 'pytorch', 'torchscript']

#Default command settings

batch_size_all = [1, 4, 8, 16, 32]
sequence_length_all = [20, 32, 128, 384, 512]
benchmark_duration_default = [60]
benchmark_duration_long = [120]
benchmark_duration_short = [10]
warmup_runs_default = [5]
command_prefix = 'PYTHONPATH=src python3'
main_prefix = '-- src/main.py --multirun'
launcher_prefix = 'launcher.py --multi_instance'
oob_command_prefix = 'PYTHONPATH=src python3 -- src/main.py --multirun'
launcher_command_prefix = 'PYTHONPATH=src python3 launcher.py --multi_instance'
kmp_affinity_default = 'verbose,granularity=fine,compact,1,0'
backends = pt_experiment_backends
backend_specific_knobs = ''
tf_specific_knobs = 'backend.num_interops_threads=1'
enable_iomp_default = [True, False]
malloc_default = ['use_default_allocator', 'enable_tcmalloc', 'enable_jemalloc']

#Read in command line args

parser = ArgumentParser("Hugging Face Model Benchmarking")
parser.add_argument("--dryrun", action='store_true', help="Prints out only the command lines and does not execute")
parser.add_argument("--backend-list", choices=BACKEND_SET_CHOICES, help="Select predetermined backend list")
args = parser.parse_args()
print(args.backend_list)
if args.backend_list == 'pt':
    backends = pt_experiment_backends
if args.backend_list == 'tf':
    backends = tf_experiment_backends
    backend_specific_knobs = tf_specific_knobs
if args.backend_list == 'pcl':
    backends = pcl_experiment_backends
if args.backend_list == 'bert_cpp':
    backends = bert_cpp_experiment_backends

#Define the experiments

oob_experiments = [{
            'name' : 'oob_experiments',
            'launcher_knobs' : {},
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : batch_size_all,
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }]

bs1_experiments = [
        {
            'name' : 'bs1_experiments',
            'launcher_knobs' : {
                'ninstances' : [1, 2, 4, 5, 10, 20, 40, 80],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [1],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }
]

bs4_batch_size_scaling_experiments = [
        {
            'name' : 'bs4_experiments_bss_inst1',
            'launcher_knobs' : {
                'ninstances' : [1],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [4],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs4_experiments_bss_inst2',
            'launcher_knobs' : {
                'ninstances' : [2],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [2],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs4_experiments_bss_inst4',
            'launcher_knobs' : {
                'ninstances' : [4],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [1],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }
]

bs8_batch_size_scaling_experiments = [
        {
            'name' : 'bs8_experiments_bss_inst1',
            'launcher_knobs' : {
                'ninstances' : [1],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [8],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs8_experiments_bss_inst2',
            'launcher_knobs' : {
                'ninstances' : [2],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [4],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs8_experiments_bss_inst4',
            'launcher_knobs' : {
                'ninstances' : [4],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [2],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs8_experiments_bss_inst8',
            'launcher_knobs' : {
                'ninstances' : [8],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [1],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }
]

bs16_batch_size_scaling_experiments = [
        {
            'name' : 'bs16_experiments_bss_inst1',
            'launcher_knobs' : {
                'ninstances' : [1],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [16],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs16_experiments_bss_inst2',
            'launcher_knobs' : {
                'ninstances' : [2],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [8],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs16_experiments_bss_inst4',
            'launcher_knobs' : {
                'ninstances' : [4],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [4],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs16_experiments_bss_inst8',
            'launcher_knobs' : {
                'ninstances' : [8],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [2],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs16_experiments_bss_inst16',
            'launcher_knobs' : {
                'ninstances' : [16],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [1],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }
]

bs32_batch_size_scaling_experiments = [
        {
            'name' : 'bs32_experiments_bss_inst1',
            'launcher_knobs' : {
                'ninstances' : [1],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [32],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs32_experiments_bss_inst2',
            'launcher_knobs' : {
                'ninstances' : [2],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [16],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs32_experiments_bss_inst4',
            'launcher_knobs' : {
                'ninstances' : [4],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [8],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs32_experiments_bss_inst8',
            'launcher_knobs' : {
                'ninstances' : [8],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [4],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        },
        {
            'name' : 'bs32_experiments_bss_inst16',
            'launcher_knobs' : {
                'ninstances' : [16],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [2],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }
]

core_count_scaling_experiments = [
        {
            'name' : 'ccs_experiments',
            'launcher_knobs' : {
                'ninstances' : [1, 2, 4, 5, 10, 20],
                'kmp_affinity' : kmp_affinity_default,
                'enable_iomp' : enable_iomp_default,
                'malloc' : malloc_default
             },
            'main_knobs' : {
                'backend' : backends,
                'batch_size' : [4, 8, 16, 32],
                'sequence_length' : sequence_length_all,
                'benchmark_duration' : benchmark_duration_default,
                'warmup_runs' : warmup_runs_default
            }
        }
]

experiment_list = [
#        oob_experiments,
        bs1_experiments,
        bs4_batch_size_scaling_experiments,
        bs8_batch_size_scaling_experiments,
        bs16_batch_size_scaling_experiments,
        bs32_batch_size_scaling_experiments,
        core_count_scaling_experiments
]


#Run the experiments

done_experiments = []
try:
    with open('done_cmds.txt', 'r') as f:
        #done_experiments = f.readlines()
        done_experiments = f.read().splitlines()
except:
    pass

print("Done:\n", done_experiments, "\n----")
for experiments in experiment_list:
    for experiment in experiments:
        print(f"Running experiment: {experiment['name']}")
        find_best_setup.hpo(
            find_best_setup.TuningMode.LATENCY,
            launcher_parameters=experiment["launcher_knobs"],
            main_parameters=experiment["main_knobs"],
            exp_name=experiment["name"]
        )
        # commands = []
        # command = command_prefix
        # launcher_knobs = experiment['launcher_knobs']
        # if len(launcher_knobs) > 0 :
        #     command += ' ' + launcher_prefix
        #     commands.append(command)
        #     for key in launcher_knobs.keys():
        #         #print(key)
        #         new_knobs = []
        #         if key != 'enable_iomp' and key != 'malloc' and key != 'ninstances':
        #             new_knob = "--" + key + '=' + str(launcher_knobs[key])
        #             new_knobs.append(new_knob)
        #         if key == 'ninstances':
        #             instance_list = launcher_knobs['ninstances']
        #             for instance in instance_list:
        #                 new_knob = "--" + key + '=' + str(instance)
        #                 new_knobs.append(new_knob)
        #         if key == 'enable_iomp':
        #             iomp_list = launcher_knobs['enable_iomp']
        #             for val in iomp_list:
        #                 if val == True:
        #                     new_knob = '--enable_iomp'
        #                 else:
        #                     new_knob = ''
        #                 new_knobs.append(new_knob)
        #         if key == 'malloc':
        #             malloc_list = launcher_knobs['malloc']
        #             for val in malloc_list:
        #                 new_knob = '--' + val
        #                 new_knobs.append(new_knob)
        #         #print(new_knobs)
        #         new_commands = []
        #         for command in commands:
        #             for new_knob in new_knobs:
        #                 new_command = command + ' ' + new_knob if new_knob != '' else command
        #                 new_commands.append(new_command)
        #         #print(new_commands)
        #         commands = new_commands
        # else:
        #     commands.append(command)

        # for command in commands:
        #     main_knobs = experiment['main_knobs']
        #     command += ' ' + main_prefix
        #     for key in main_knobs.keys():
        #         #print(key)
        #         command += ' ' + key + '=' + ','.join([str(elem) for elem in main_knobs[key]])
        #     command += ' ' + backend_specific_knobs
        #     command += ' experiment_name=' + experiment['name']
        #     #command += ' hydra.run.dir=outputs/' + experiment['name'] + ' hydra.sweep.dir=outputs/' + experiment['name'];
        #     #command += ' hydra.run.dir=outputs/' + experiment['name'] + '/${experiment_id}/${instance_id}' + ' hydra.sweep.dir=outputs/' + experiment['name'] + '/${experiment_id}/${instance_id}';
        #     print("###################################################################################################")
        #     print(command)
            # if command in done_experiments:
            #     print("Skipping...")
            #     continue
            # if not args.dryrun:
            #     #print("executing ----")
            #     try:
            #         subprocess.run(command, check=True, text=True, shell=True, executable='/bin/bash')
            #         with open('done_cmds.txt', 'a') as f:
            #             print(command, file=f)
            #     except Exception as e:
            #         print(e)
            #         print(f"FAILED: {command}")
            #         with open('failed_cmds.txt', 'a') as f:
            #             print(command, file=f)
