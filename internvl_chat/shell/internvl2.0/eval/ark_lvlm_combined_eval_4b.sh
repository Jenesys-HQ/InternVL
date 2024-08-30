set -x

CURRENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")
MODEL_NAME="ark_lvlm_lora_4b"
MODEL_PATH="models/${MODEL_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export MLFLOW_TRACKING_ARN="arn:aws:sagemaker:eu-west-1:899757773314:mlflow-tracking-server/test"
export RUN_NAME="${MODEL_NAME}_eval_${CURRENT_DATE}"

python tools/eval.py \
  --model-path ${MODEL_PATH} \
  --eval-dataset "data/processed_whole/ark-lvlm-combined/test.jsonl" \
  2>&1 | tee -a "${MODEL_PATH}/${CURRENT_DATE}_eval_log.txt"
