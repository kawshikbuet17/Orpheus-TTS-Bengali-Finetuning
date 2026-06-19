conda activate orpheus-tts



cd /home/kawshik/tts-testing/Orpheus-TTS



python - <<'PY'
from datasets import load_dataset

ds = load_dataset("kawshikbuet17/bengali-telecom-customer-care-speech")
print(ds)
print(ds["train"].column_names)
print(ds["train"][0]["text"])
print(ds["train"][0]["text_normalized"])
print(ds["train"][0]["audio"])
PY



mkdir -p data


vi preprocess_bengali_telecom_orpheus_local.py


CUDA_VISIBLE_DEVICES=1 python preprocess_bengali_telecom_orpheus_local.py


python - <<'PY'
from datasets import load_from_disk

path = "/home/kawshik/tts-testing/Orpheus-TTS/data/bengali-telecom-orpheus-tokenized"

ds = load_from_disk(path)

print(ds)
print(ds["train"].column_names)
print(ds["train"][0].keys())

required = {"input_ids", "labels", "attention_mask"}
missing = required - set(ds["train"].column_names)

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

print("OK")
print("train rows:", len(ds["train"]))
print("validation rows:", len(ds["validation"]) if "validation" in ds else "no validation")
print("test rows:", len(ds["test"]) if "test" in ds else "no test")
print("first input length:", len(ds["train"][0]["input_ids"]))
PY



cd /home/kawshik/tts-testing/Orpheus-TTS/finetune

vi config_bengali_full.yaml

vi train_full_bengali_local.py


cd /home/kawshik/tts-testing/Orpheus-TTS/finetune

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 accelerate launch train_full_bengali_local.py


Failed

pip install -U accelerate wandb

Then run again


cd /home/kawshik/tts-testing/Orpheus-TTS/finetune

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 python -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  train_full_bengali_local.py


Failed because of OOM. Should try LORA


pip install peft


python - <<'PY'
import torch
import transformers
import accelerate
import peft
import bitsandbytes as bnb

print("python ok")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("bitsandbytes:", bnb.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY


vi config_bengali_qlora.yaml

vi train_qlora_bengali_local.py


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



vi infer_qlora_checkpoint_bengali.py


