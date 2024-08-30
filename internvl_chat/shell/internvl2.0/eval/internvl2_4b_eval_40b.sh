set -x

CURRENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")
RUN_NAME="internvl2_40b_eval_${CURRENT_DATE}"
OUTPUT_DIR="eval/${RUN_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export MLFLOW_TRACKING_ARN="arn:aws:sagemaker:eu-west-1:899757773314:mlflow-tracking-server/test"

python tools/eval.py \
  --model-path "OpenGVLab/InternVL2-40B" \
  --eval-dataset "data/processed_whole/ark-lvlm-combined/test.jsonl" \
  2>&1 | tee -a "${OUTPUT_DIR}/${CURRENT_DATE}_eval_log.txt"
