START_PORT=${START_PORT:-29165}
MASTER_PORT=${MASTER_PORT:-29175}
GPU_ID=${GPU_ID:-4}

save_root='visualization/'
mkdir -p $save_root

CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root


