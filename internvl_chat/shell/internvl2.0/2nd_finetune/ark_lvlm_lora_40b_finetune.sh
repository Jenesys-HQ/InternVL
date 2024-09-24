set -x

CURRENT_DATE=$(date +"%Y-%m-%d-%H-%M-%S")
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
EPOCHS=${EPOCHS:-1}
DATASET_NAME=$1
MODEL_NAME="ark_lvlm_lora_40b_${DATASET_NAME}"
RUN_NAME="${MODEL_NAME}_finetune_${CURRENT_DATE}"
OUTPUT_DIR="models/${RUN_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export HF_MLFLOW_LOG_ARTIFACTS="true"
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:eu-west-1:899757773314:mlflow-tracking-server/test"
export MLFLOW_EXPERIMENT_NAME="Invoice Extraction"
export MFLOW_FLATTEN_PARAMS="true"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/InternVL2-40B" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/${DATASET_NAME}_train.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config_34b.json" \
  --report_to "mlflow" \
  --run_name "${RUN_NAME}" \
  2>&1 | tee -a "${OUTPUT_DIR}/${CURRENT_DATE}_training_log.txt"
