START_PORT=${1:-29556}
MASTER_PORT=${2:-29661}
GPU_OFFSET=${3:-0}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root='./visualization/'
mkdir -p $save_root

batch_time=$(date +%Y%m%d_%H%M%S)


num_gpus=2
for i in $(seq 0 $((num_gpus - 1))); do
    gpu_id=$(( i + GPU_OFFSET ))
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${gpu_id}_${batch_time}.log"
    echo "[Task $i] GPU: ${gpu_id} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${gpu_id}  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --save_root $save_root \
        --port $CURRENT_PORT  > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All instances have been launched in the background."
wait
