#!/bin/bash
#
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

BS_LIST="1 4 8 16 32"
SEQ_LEN="20 32 128 384 512"

helper() {
   echo ""
   echo "Usage: $0 <-f framework> <-m mode> <-t trials_per_cfg> [-p proxy] [-r]"
   echo -e "\t-f: framework to be tested, [pytorch, torchscript]"
   echo -e "\t-m: benchmark mode, 0: latency, 1: throughput"
   echo -e "\t-t: trials budget per configuration, recommend >= 50"
   echo -e "\t-p: network proxy to reach sigopt if needed"
   echo -e "\t-r: try run - to dump the run command only"
   echo ""
   exit 1
}

proxy="None"
execute_cmd=1
# step 1: check cmdline option, : meaning a parameter is needed for that option
while getopts "f:m:t:p:r" opt
do
   case "$opt" in
      f ) framework="$OPTARG" ;;
      m ) mode="$OPTARG" ;;
      t ) trials_per_cfg="$OPTARG" ;;
      p ) proxy="$OPTARG" ;;
      r ) execute_cmd=0 ;;
      ? ) helper ;; # echo helper in case parameter is non-existent
   esac
done

# echo helper in case parameters are empty
if [ -z "$framework" ] || [ -z "$trials_per_cfg" ] || [ -z "$mode" ]
then
   echo -e "\n!!! WRONG INPUT PARAMS !!!";
   helper
fi

# step 2: launch tuning jobs
for bs in $BS_LIST; do
  for seq in $SEQ_LEN; do
    cmd="PYTHONPATH=src python sigopt_tune.py --framework $framework --mode $mode --proxy $proxy --batch_size $bs
         --sequence_length $seq --n_trials $trials_per_cfg"
    echo $cmd
    if [ $execute_cmd -eq 1 ]; then eval $cmd; fi
  done
done

# step 3: parse the result
if [ $execute_cmd -eq 1 ]; then PYTHONPATH=src python sigopt_tune.py --convert_csv --logfile tune.log; fi

echo -e "\n!!! DONE FOR THE ENTIRE JOB !!!\n"
