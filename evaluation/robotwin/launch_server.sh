START_PORT=${START_PORT:-29058}
MASTER_PORT=${MASTER_PORT:-29063}
GPU_ID=${GPU_ID:-7}

save_root='visualization/'
mkdir -p $save_root

CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root


