import os
import gc
import torch
import torchaudio.transforms as T

from datasets import load_dataset, Audio, DatasetDict, Dataset
from transformers import AutoTokenizer
from snac import SNAC


SOURCE_DATASET = "kawshikbuet17/bengali-telecom-customer-care-speech"
TOKENIZER_NAME = "canopylabs/orpheus-3b-0.1-pretrained"

OUT_DIR = "/home/kawshik/tts-testing/Orpheus-TTS/data/bengali-telecom-orpheus-tokenized"

TARGET_SR = 24000

# Keep punctuation for TTS prosody.
TEXT_FIELD = "text"

# First run can be tested with a small number.
# Set to None for full dataset.
MAX_ROWS_PER_SPLIT = None
# MAX_ROWS_PER_SPLIT = 50

# If one sample becomes too long, skip it.
# Your max duration is around 17.8s, so 8192 is usually okay. but using 4096 to be safe and avoid OOMs.
MAX_LEN = 4096

TOKENIZER_LENGTH = 128256

START_OF_TEXT = 128000
END_OF_TEXT = 128009

START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2

START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4

START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6

PAD_TOKEN = TOKENIZER_LENGTH + 7

AUDIO_TOKEN_START = TOKENIZER_LENGTH + 10  # 128266

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_text(example):
    text = example.get(TEXT_FIELD) or example.get("text_normalized") or ""
    text = text.strip()
    text = " ".join(text.split())
    return text


def encode_audio_to_snac_tokens(snac_model, audio_dict):
    audio = audio_dict["array"]
    sr = audio_dict["sampling_rate"]

    waveform = torch.tensor(audio, dtype=torch.float32)

    # Convert multi-channel to mono if needed.
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    waveform = waveform.unsqueeze(0)  # [1, T]

    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    audio_tokens = []

    # Orpheus/SNAC interleaving pattern.
    # 1 token from level 0, 2 from level 1, 4 from level 2 = 7 tokens per frame.
    for i in range(codes[0].shape[1]):
        audio_tokens.append(codes[0][0][i].item() + AUDIO_TOKEN_START)
        audio_tokens.append(codes[1][0][2 * i].item() + AUDIO_TOKEN_START + 4096)
        audio_tokens.append(codes[2][0][4 * i].item() + AUDIO_TOKEN_START + 2 * 4096)
        audio_tokens.append(codes[2][0][4 * i + 1].item() + AUDIO_TOKEN_START + 3 * 4096)
        audio_tokens.append(codes[1][0][2 * i + 1].item() + AUDIO_TOKEN_START + 4 * 4096)
        audio_tokens.append(codes[2][0][4 * i + 2].item() + AUDIO_TOKEN_START + 5 * 4096)
        audio_tokens.append(codes[2][0][4 * i + 3].item() + AUDIO_TOKEN_START + 6 * 4096)

    return audio_tokens


def remove_duplicate_frames(tokens):
    if not tokens:
        return tokens

    if len(tokens) % 7 != 0:
        raise ValueError(f"SNAC token length must be divisible by 7, got {len(tokens)}")

    result = tokens[:7]

    for i in range(7, len(tokens), 7):
        current_first = tokens[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(tokens[i:i + 7])

    return result


def make_input_ids(tokenizer, text, audio_tokens):
    text_ids = tokenizer.encode(text, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)

    input_ids = (
        [START_OF_HUMAN]
        + text_ids
        + [END_OF_HUMAN]
        + [START_OF_AI]
        + [START_OF_SPEECH]
        + audio_tokens
        + [END_OF_SPEECH]
        + [END_OF_AI]
    )

    return input_ids


def process_split(split_name, split_ds, tokenizer, snac_model):
    if MAX_ROWS_PER_SPLIT is not None:
        split_ds = split_ds.select(range(min(MAX_ROWS_PER_SPLIT, len(split_ds))))

    processed_rows = []
    total = len(split_ds)

    print(f"\nProcessing split={split_name}, rows={total}")

    for idx, example in enumerate(split_ds):
        try:
            text = get_text(example)

            if not text:
                continue

            audio_tokens = encode_audio_to_snac_tokens(snac_model, example["audio"])
            audio_tokens = remove_duplicate_frames(audio_tokens)

            if not audio_tokens:
                continue

            input_ids = make_input_ids(tokenizer, text, audio_tokens)

            if len(input_ids) > MAX_LEN:
                print(f"Skipping long sample split={split_name} idx={idx}, len={len(input_ids)}")
                continue

            processed_rows.append({
                "input_ids": input_ids,
                "labels": input_ids.copy(),
                "attention_mask": [1] * len(input_ids),
            })

        except Exception as e:
            print(f"Skipping split={split_name} idx={idx} due to error: {repr(e)}")

        if idx % 100 == 0:
            print(f"{split_name}: {idx}/{total}, kept={len(processed_rows)}")

        if idx % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return Dataset.from_list(processed_rows)


def main():
    print("Device:", device)
    print("Source dataset:", SOURCE_DATASET)
    print("Text field:", TEXT_FIELD)

    print("Loading raw dataset...")
    raw = load_dataset(SOURCE_DATASET)

    # Dataset already says 24k, but this guarantees decoded audio is 24k.
    raw = raw.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    print(raw)

    print("Loading Orpheus tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading SNAC 24kHz model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(device)
    snac_model.eval()

    out = DatasetDict()

    for split_name in raw.keys():
        out[split_name] = process_split(
            split_name=split_name,
            split_ds=raw[split_name],
            tokenizer=tokenizer,
            snac_model=snac_model,
        )

    print("\nTokenized dataset:")
    print(out)

    os.makedirs(OUT_DIR, exist_ok=True)
    out.save_to_disk(OUT_DIR)

    print("\nSaved local tokenized dataset:")
    print(OUT_DIR)

    print("\nExample check:")
    print(out["train"][0].keys())
    print("input_ids length:", len(out["train"][0]["input_ids"]))


if __name__ == "__main__":
    main()