set -x

CURRENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")
GPUS=${GPUS:-1}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export MLFLOW_TRACKING_ARN="arn:aws:sagemaker:eu-west-1:899757773314:mlflow-tracking-server/test"

OUTPUT_DIR="./pretrained/InternVL2-4B"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  tools/eval.py \
  --model-path ${OUTPUT_DIR} \
  --eval-dataset "data/processed_whole/ark-lvlm-combined/test.jsonl" \
  2>&1 | tee -a "${OUTPUT_DIR}/${CURRENT_DATE}_eval_log.txt"
