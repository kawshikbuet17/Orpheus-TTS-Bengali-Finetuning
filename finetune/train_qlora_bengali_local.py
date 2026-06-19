import os
import inspect
import yaml
import torch
import wandb

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


CONFIG_FILE = "config_bengali_qlora.yaml"


class CausalLMPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            cur_len = len(x["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)

            # Ignore padding tokens in loss.
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_training_args(**kwargs):
    """
    Your transformers version rejected some TrainingArguments fields earlier.
    This helper removes unsupported args automatically.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}

    skipped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if skipped:
        print("Skipped unsupported TrainingArguments:", skipped)

    return TrainingArguments(**filtered)


def main():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    dataset_path = config["TTS_dataset"]
    model_name = config["model_name"]
    output_dir = config["save_folder"]

    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    grad_accum = int(config.get("gradient_accumulation_steps", 4))
    save_steps = int(config["save_steps"])
    learning_rate = float(config["learning_rate"])
    pad_token = int(config.get("pad_token", 128263))

    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    lora_dropout = float(config.get("lora_dropout", 0.05))

    project_name = config.get("project_name", "orpheus-qlora-local")
    run_name = config.get("run_name", "qlora-local")

    print("Dataset path:", dataset_path)
    print("Base model:", model_name)
    print("Output dir:", output_dir)

    ds = load_from_disk(dataset_path)

    if "train" not in ds:
        raise ValueError("Dataset must contain train split.")

    train_ds = ds["train"]
    eval_ds = ds["validation"] if "validation" in ds else None

    required = {"input_ids", "labels", "attention_mask"}
    missing = required - set(train_ds.column_names)
    if missing:
        raise ValueError(f"Tokenized dataset missing columns: {missing}")

    print("Train rows:", len(train_ds))
    if eval_ds is not None:
        print("Validation rows:", len(eval_ds))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token

    print("Loading base model in 4-bit QLoRA mode...")

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print("compute_dtype:", compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False

    # Important for k-bit training.
    model = prepare_model_for_kbit_training(model)

    # Llama-style target modules.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(project=project_name, name=run_name)

    training_args = make_training_args(
        output_dir=output_dir,

        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        learning_rate=learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,

        logging_steps=1,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,

        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),

        # Paged optimizer helps QLoRA memory spikes.
        optim="paged_adamw_8bit",

        report_to="wandb",
        remove_unused_columns=False,

        dataloader_num_workers=1,
    )

    collator = CausalLMPaddingCollator(pad_token_id=pad_token)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "adapter-final")
    print("Saving LoRA adapter to:", final_dir)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done.")
    print("Adapter saved at:", final_dir)


if __name__ == "__main__":
    main()