import os
import yaml
import torch
import wandb

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


CONFIG_FILE = "config_bengali_full.yaml"


class CausalLMPaddingCollator:
    def __init__(self, pad_token_id):
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

            # Ignore pad tokens in loss.
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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

    project_name = config.get("project_name", "orpheus-local")
    run_name = config.get("run_name", "full-finetune-local")

    print("Dataset path:", dataset_path)
    print("Base model:", model_name)
    print("Output dir:", output_dir)

    ds = load_from_disk(dataset_path)

    if "train" not in ds:
        raise ValueError("Local dataset must contain a train split.")

    train_ds = ds["train"]
    eval_ds = ds["validation"] if "validation" in ds else None

    required = {"input_ids", "labels", "attention_mask"}
    missing = required - set(train_ds.column_names)

    if missing:
        raise ValueError(
            f"Dataset is not Orpheus-tokenized. Missing: {missing}. "
            f"Current columns: {train_ds.column_names}"
        )

    print("Train rows:", len(train_ds))
    if eval_ds is not None:
        print("Validation rows:", len(eval_ds))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token

    print("Loading model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        print("Loaded with flash_attention_2")
    except Exception as e:
        print("flash_attention_2 failed. Loading without it.")
        print("Reason:", repr(e))
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(project=project_name, name=run_name)

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        learning_rate=learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_steps=1,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,

        fp16=False,
        bf16=True,

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

    final_dir = os.path.join(output_dir, "final")
    print("Saving final model to:", final_dir)

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Done.")
    print("Final model:", final_dir)


if __name__ == "__main__":
    main()