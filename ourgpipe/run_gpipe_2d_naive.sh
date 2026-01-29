#!/bin/bash
# Node0 (1 GPUs)
# if [ $HOSTNAME == "paraai-n32-h-01-agent-16" ]; then
# nsys profile -o gpipe_stream --force-overwrite true \
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr="127.0.0.1" \
    --master_port=1231 \
    gpipe_thread-stream2.py

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