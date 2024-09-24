set -x

CURRENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")
DATASET_NAME=$1
MODEL_NAME="ark_lvlm_lora_26b_${DATASET_NAME}"
MODEL_PATH="models/${MODEL_NAME}"

if [ ! -d "$MODEL_PATH" ]; then
  mkdir -p "$MODEL_PATH"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:eu-west-1:899757773314:mlflow-tracking-server/test"
export MLFLOW_EXPERIMENT_NAME="Invoice Extraction"
export MLFLOW_RUN_NAME="${MODEL_NAME}_eval_${CURRENT_DATE}"

python tools/eval.py \
  --model-path ${MODEL_PATH} \
  --eval-dataset "data/processed_whole/${DATASET_NAME}/test.jsonl" \
  2>&1 | tee -a "${MODEL_PATH}/${CURRENT_DATE}_eval_log.txt"
