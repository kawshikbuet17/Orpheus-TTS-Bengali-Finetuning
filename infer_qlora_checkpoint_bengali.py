import sys
import torch
import soundfile as sf

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from snac import SNAC


BASE_MODEL = "canopylabs/orpheus-3b-0.1-pretrained"

TOKENIZER_LENGTH = 128256

END_OF_TEXT = 128009

START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2

START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4

START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6

PAD_TOKEN = TOKENIZER_LENGTH + 7

AUDIO_TOKEN_START = TOKENIZER_LENGTH + 10
AUDIO_TOKEN_END = AUDIO_TOKEN_START + 7 * 4096

SAMPLE_RATE = 24000


def build_prompt_ids(tokenizer, text: str):
    text_ids = tokenizer.encode(text, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)

    return (
        [START_OF_HUMAN]
        + text_ids
        + [END_OF_HUMAN]
        + [START_OF_AI]
        + [START_OF_SPEECH]
    )


def extract_audio_tokens(generated_ids):
    audio_tokens = []

    for token in generated_ids:
        token = int(token)

        if token in [END_OF_SPEECH, END_OF_AI, PAD_TOKEN]:
            break

        if AUDIO_TOKEN_START <= token < AUDIO_TOKEN_END:
            audio_tokens.append(token)

    usable_len = (len(audio_tokens) // 7) * 7
    return audio_tokens[:usable_len]


def audio_tokens_to_waveform(snac_model, audio_tokens, device):
    if len(audio_tokens) == 0:
        raise RuntimeError("No audio tokens generated.")

    if len(audio_tokens) % 7 != 0:
        raise RuntimeError(f"Audio token length must be divisible by 7, got {len(audio_tokens)}")

    n_frames = len(audio_tokens) // 7

    codes_0 = []
    codes_1 = []
    codes_2 = []

    for i in range(n_frames):
        t = audio_tokens[i * 7 : (i + 1) * 7]

        c0 = t[0] - AUDIO_TOKEN_START

        c1_0 = t[1] - AUDIO_TOKEN_START - 4096
        c2_0 = t[2] - AUDIO_TOKEN_START - 2 * 4096
        c2_1 = t[3] - AUDIO_TOKEN_START - 3 * 4096
        c1_1 = t[4] - AUDIO_TOKEN_START - 4 * 4096
        c2_2 = t[5] - AUDIO_TOKEN_START - 5 * 4096
        c2_3 = t[6] - AUDIO_TOKEN_START - 6 * 4096

        vals = [c0, c1_0, c2_0, c2_1, c1_1, c2_2, c2_3]
        if any(v < 0 or v >= 4096 for v in vals):
            continue

        codes_0.append(c0)
        codes_1.extend([c1_0, c1_1])
        codes_2.extend([c2_0, c2_1, c2_2, c2_3])

    if not codes_0:
        raise RuntimeError("No valid SNAC frames found from generated tokens.")

    codes = [
        torch.tensor([codes_0], dtype=torch.long, device=device),
        torch.tensor([codes_1], dtype=torch.long, device=device),
        torch.tensor([codes_2], dtype=torch.long, device=device),
    ]

    with torch.inference_mode():
        audio = snac_model.decode(codes)

    return audio.detach().squeeze().float().cpu().numpy()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python infer_qlora_checkpoint_bengali.py <adapter_path> [prompt] [output.wav]")
        sys.exit(1)

    adapter_path = sys.argv[1]

    prompt = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else "আপনার রকেট অ্যাকাউন্টের লাস্ট রিচার্জ কত ছিল, বলতে পারবেন?"
    )

    out_path = sys.argv[3] if len(sys.argv) >= 4 else "qlora_checkpoint_test.wav"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print("Base model:", BASE_MODEL)
    print("Adapter:", adapter_path)
    print("Prompt:", prompt)
    print("Output:", out_path)
    print("Device:", device)
    print("Compute dtype:", compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading base model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Loading QLoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("Loading SNAC decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    snac_model.eval()

    input_ids_list = build_prompt_ids(tokenizer, prompt)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    print("Generating speech tokens...")

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=END_OF_SPEECH,
            pad_token_id=PAD_TOKEN,
        )

    generated_new_tokens = output[0][input_ids.shape[1]:].tolist()
    audio_tokens = extract_audio_tokens(generated_new_tokens)

    print("Generated new tokens:", len(generated_new_tokens))
    print("Audio tokens:", len(audio_tokens))
    print("Audio frames:", len(audio_tokens) // 7)

    audio = audio_tokens_to_waveform(snac_model, audio_tokens, device)

    sf.write(out_path, audio, SAMPLE_RATE)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()