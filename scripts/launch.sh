#!/bin/bash

KMP_AFF=verbose,granularity=fine,compact,1,0
BACKEND=pytorch,ov,tensorflow
LOG_BK=pt-ov-tf
BATCH_SIZE=1,2,4,8
SEQ_LEN=20,32,128,384,512
BENCH_DURATION=20
WARMUP_RUN=5

N_INSTS=(1 2 4 8)
MODELS=(bert-base-cased distilbert-base-uncased gpt2)

for MODEL in ${MODELS[@]}; do
    for N_INST in ${N_INSTS[@]}; do
        cmd_to_run="PYTHONPATH=src python3 launcher.py \
        --multi_instance \
        --ninstances=$N_INSTS \
        --kmp_affinity=$KMP_AFF \
        --enable_iomp \
        --enable_tcmalloc \
        -- src/main.py \
        --multirun \
        backend=$BACKEND \
        batch_size=$BATCH_SIZE \
        sequence_length=$SEQ_LEN \
        benchmark_duration=$BENCH_DURATION \
        warmup_runs=$WARMUP_RUN \
        model=$MODEL \
        2>&1 | \
        tee logs/$MODEL-$LOG_BK-instances-$N_INST-dur-$BENCH_DURATION-wup-$WARMUP_RUN.log "

        echo "*** Starting benchmark with :"
        echo $cmd_to_run
        eval $cmd_to_run
    done
done;
