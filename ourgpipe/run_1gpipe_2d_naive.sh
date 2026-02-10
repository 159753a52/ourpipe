#!/bin/bash
# Node0 (1 GPUs)
# if [ $HOSTNAME == "paraai-n32-h-01-agent-16" ]; then
# nsys profile -o gpipe_stream --force-overwrite true \
export TORCHINDUCTOR_CACHE_DIR=/home/bingxing2/home/scx9kvs/mxy/GPipe/ourgpipe/.torchinductor_cache
export TORCHINDUCTOR_DEVICE_CACHE_DIR=/home/bingxing2/home/scx9kvs/mxy/GPipe/ourgpipe/.torchinductor_device_cache
export TORCHINDUCTOR_COMPILE_THREADS=128      # 多线程加速编译

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr="127.0.0.1" \
    --master_port=1231 \
    gpipe_naive.py

# Node1 (1 GPU)
# elif [ $HOSTNAME == "paraai-n32-h-01-agent-16" ]; then
#     torchrun \
#         --nnodes=4 \
#         --node_rank=1 \
#         --nproc_per_node=1 \
#         --master_addr="127.0.0.1" \
#         --master_port=1231 \
#         gpipe_2d.py

# # Node2 (1 GPU)
# elif [ $HOSTNAME == "paraai-n32-h-01-agent-16" ]; then
#     torchrun \
#         --nnodes=4 \
#         --node_rank=2 \
#         --nproc_per_node=1 \
#         --master_addr="127.0.0.1" \
#         --master_port=1231 \
#         gpipe_2d.py

# # Node3 (1 GPU)
# elif [ $HOSTNAME == "paraai-n32-h-01-agent-16" ]; then
#     torchrun \
#         --nnodes=4 \
#         --node_rank=3 \
#         --nproc_per_node=1 \
#         --master_addr="127.0.0.1" \
#         --master_port=1231 \
#         gpipe_2d.py
# fi