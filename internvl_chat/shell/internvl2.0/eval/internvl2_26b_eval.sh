set -x

CURRENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")
MODEL_NAME="internvl2_40b"
OUTPUT_DIR="models/${MODEL_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:eu-west-1:899757773314:mlflow-tracking-server/test"
export MLFLOW_EXPERIMENT_NAME="Invoice Extraction"
export RUN_NAME="${MODEL_NAME}_eval_${CURRENT_DATE}"

python tools/eval.py \
  --model-path "OpenGVLab/InternVL2-40B" \
  --eval-dataset "data/processed_whole/ark-lvlm-combined/test.jsonl" \
  2>&1 | tee -a "${OUTPUT_DIR}/${CURRENT_DATE}_eval_log.txt"
