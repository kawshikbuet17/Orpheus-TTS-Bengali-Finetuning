cd /home/kawshik/tts-testing/Orpheus-TTS
CUDA_VISIBLE_DEVICES=1 python preprocess_bengali_telecom_orpheus_local.py

cd /home/kawshik/tts-testing/Orpheus-TTS/finetune

rm -rf checkpoints-bengali-telecom-qlora

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 python -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_qlora_bengali_local.py \
  > train_qlora_bengali_local.log
