import os
import sys
import time
import wave
import asyncio
from pathlib import Path
from statistics import mean

import gradio as gr
from vllm import SamplingParams

# Use cloned repository-local package.
# This avoids the PyPI max_model_len issue.
sys.path.insert(0, "orpheus_tts_pypi")

from orpheus_tts import OrpheusModel
from orpheus_tts.decoder import tokens_decoder


# -----------------------------
# Paths / folders
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
BENCHMARK_OUTPUT_DIR = BASE_DIR / "benchmark_outputs" / "ui_seq_vs_concurrent"
GRADIO_TMP_DIR = BASE_DIR / "gradio_tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Keep Gradio temp files inside project folder.
os.environ["TMPDIR"] = str(GRADIO_TMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP_DIR)


# -----------------------------
# Model config
# -----------------------------
MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"

VOICE_OPTIONS = [
    "tara",
    "leah",
    "jess",
    "leo",
    "dan",
    "mia",
    "zac",
    "zoe",
]

DEFAULT_VOICE = "tara"
SAMPLE_RATE = 24000

MAX_MODEL_LEN = 1024
GPU_MEMORY_UTILIZATION = 0.45

MAX_UI_AUDIOS = 4

DEFAULT_TEXTS = """Hello, this is the first Orpheus model level batching test. We are checking whether the first request can generate complete audio while other requests are active at the same time.
This is the second request, submitted concurrently to the vLLM engine. The goal is to verify that asynchronous scheduling can process multiple speech token streams without cutting the audio short.
The third request checks whether model level batching improves wall time. We will compare sequential generation with concurrent generation and also verify that the output duration is not suspiciously tiny.
The fourth request measures total wall time and audio throughput. If batching works correctly, concurrent wall time should be lower while the generated audio should remain complete and playable."""


# -----------------------------
# Global model
# -----------------------------
model = None


def load_model_once():
    global model

    if model is not None:
        return model

    print("Loading Orpheus-TTS model once...")
    print(f"Model: {MODEL_NAME}")
    print("Using local package: orpheus_tts_pypi")
    print(f"max_model_len={MAX_MODEL_LEN}")
    print(f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")

    model = OrpheusModel(
        model_name=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    print("Model loaded.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Benchmark output directory: {BENCHMARK_OUTPUT_DIR}")
    print(f"Gradio temp directory: {GRADIO_TMP_DIR}")

    return model


# -----------------------------
# Helpers
# -----------------------------
def parse_texts(texts_blob: str):
    texts = []

    for line in (texts_blob or "").splitlines():
        line = line.strip()
        if line:
            texts.append(line)

    return texts


def pad_items(items, max_items=MAX_UI_AUDIOS, fill_value=None):
    items = list(items[:max_items])

    while len(items) < max_items:
        items.append(fill_value)

    return items


def write_audio_chunks_to_wav(audio_chunks, path: Path):
    total_frames = 0
    chunk_count = 0

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        for chunk in audio_chunks:
            chunk_count += 1
            total_frames += len(chunk) // 2
            wf.writeframes(chunk)

    audio_duration = total_frames / SAMPLE_RATE
    return audio_duration, chunk_count


def make_sampling_params(
    temperature: float,
    top_p: float,
    max_tokens: int,
    min_tokens: int,
    repetition_penalty: float,
):
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop_token_ids=[49158],
        repetition_penalty=repetition_penalty,
    )


async def vllm_token_stream(
    model_obj: OrpheusModel,
    text: str,
    voice: str,
    request_id: str,
    sampling_params: SamplingParams,
    logs: list[str],
):
    """
    Direct async vLLM token stream for Orpheus.

    Important:
    We do not stream result.outputs[0].text directly.
    We diff result.outputs[0].token_ids and decode each newly generated token ID
    into a token string. This was required to avoid incomplete audio in concurrent mode.
    """
    prompt_string = model_obj._format_prompt(text, voice)

    seen_token_count = 0
    event_count = 0
    yielded_token_count = 0
    fallback_text_count = 0

    async for result in model_obj.engine.generate(
        prompt=prompt_string,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        event_count += 1

        if not result.outputs:
            continue

        output = result.outputs[0]
        token_ids = getattr(output, "token_ids", None)

        if token_ids is None:
            piece = output.text
            if piece:
                fallback_text_count += 1
                yielded_token_count += 1
                yield piece
            continue

        # vLLM usually returns generated token_ids cumulatively.
        new_token_ids = token_ids[seen_token_count:]
        seen_token_count = len(token_ids)

        for token_id in new_token_ids:
            token_text = model_obj.tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if token_text:
                yielded_token_count += 1
                yield token_text

    logs.append(
        f"[{request_id}] events={event_count}, "
        f"generated_token_ids={seen_token_count}, "
        f"yielded_token_strings={yielded_token_count}, "
        f"fallback_text_chunks={fallback_text_count}"
    )


async def generate_one_async(
    model_obj: OrpheusModel,
    text: str,
    voice: str,
    idx: int,
    mode: str,
    run_dir: Path,
    sampling_params: SamplingParams,
    logs: list[str],
):
    out_path = run_dir / f"{mode}_{idx}.wav"
    request_id = f"orpheus-{mode}-{idx}-{time.time_ns()}"

    start = time.monotonic()
    audio_chunks = []

    async for audio_chunk in tokens_decoder(
        vllm_token_stream(
            model_obj=model_obj,
            text=text,
            voice=voice,
            request_id=request_id,
            sampling_params=sampling_params,
            logs=logs,
        )
    ):
        audio_chunks.append(audio_chunk)

    elapsed = time.monotonic() - start

    audio_duration, chunk_count = write_audio_chunks_to_wav(
        audio_chunks=audio_chunks,
        path=out_path,
    )

    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

    return {
        "idx": idx,
        "mode": mode,
        "text": text,
        "path": str(out_path),
        "elapsed": elapsed,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "chunk_count": chunk_count,
    }


async def warmup_async(
    model_obj: OrpheusModel,
    voice: str,
    sampling_params: SamplingParams,
    logs: list[str],
):
    logs.append("\nWarmup...")

    warmup_text = (
        "This is a short warmup request before measuring Orpheus concurrent "
        "model level batching."
    )

    chunks = []

    async for audio_chunk in tokens_decoder(
        vllm_token_stream(
            model_obj=model_obj,
            text=warmup_text,
            voice=voice,
            request_id=f"warmup-{time.time_ns()}",
            sampling_params=sampling_params,
            logs=logs,
        )
    ):
        chunks.append(audio_chunk)

    logs.append(f"Warmup done. Warmup chunks={len(chunks)}")


async def run_sequential_async(
    model_obj: OrpheusModel,
    texts: list[str],
    voice: str,
    run_dir: Path,
    sampling_params: SamplingParams,
    logs: list[str],
):
    logs.append("\n==============================")
    logs.append("Sequential async-vLLM run")
    logs.append("==============================")

    start = time.monotonic()
    results = []

    for idx, text in enumerate(texts, start=1):
        logs.append(f"\nSequential item {idx}/{len(texts)}")
        logs.append(f"Text: {text}")

        result = await generate_one_async(
            model_obj=model_obj,
            text=text,
            voice=voice,
            idx=idx,
            mode="seq",
            run_dir=run_dir,
            sampling_params=sampling_params,
            logs=logs,
        )

        results.append(result)

        logs.append(
            f"seq {idx}: elapsed={result['elapsed']:.2f}s, "
            f"audio={result['audio_duration']:.2f}s, "
            f"RTF={result['rtf']:.3f}, "
            f"chunks={result['chunk_count']}, "
            f"path={result['path']}"
        )

    wall_time = time.monotonic() - start
    total_audio = sum(r["audio_duration"] for r in results)
    throughput = total_audio / wall_time if wall_time > 0 else 0.0
    avg_rtf = mean(r["rtf"] for r in results) if results else 0.0
    effective_rtf = wall_time / total_audio if total_audio > 0 else 0.0

    logs.append("\n--- Sequential Summary ---")
    logs.append(f"Sequential wall time: {wall_time:.2f}s")
    logs.append(f"Sequential total audio: {total_audio:.2f}s")
    logs.append(f"Sequential throughput: {throughput:.2f} audio-sec/wall-sec")
    logs.append(f"Sequential effective RTF: {effective_rtf:.3f}")
    logs.append(f"Sequential avg per-request RTF: {avg_rtf:.3f}")

    return wall_time, total_audio, results


async def run_concurrent_async(
    model_obj: OrpheusModel,
    texts: list[str],
    voice: str,
    run_dir: Path,
    sampling_params: SamplingParams,
    concurrency: int,
    logs: list[str],
):
    logs.append("\n==============================")
    logs.append(f"Concurrent async-vLLM run: concurrency={concurrency}")
    logs.append("==============================")

    texts = texts[:concurrency]

    start = time.monotonic()

    tasks = [
        asyncio.create_task(
            generate_one_async(
                model_obj=model_obj,
                text=text,
                voice=voice,
                idx=idx,
                mode="concurrent",
                run_dir=run_dir,
                sampling_params=sampling_params,
                logs=logs,
            )
        )
        for idx, text in enumerate(texts, start=1)
    ]

    results = []

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)

        logs.append(
            f"concurrent {result['idx']}: elapsed={result['elapsed']:.2f}s, "
            f"audio={result['audio_duration']:.2f}s, "
            f"RTF={result['rtf']:.3f}, "
            f"chunks={result['chunk_count']}, "
            f"path={result['path']}"
        )

    wall_time = time.monotonic() - start
    total_audio = sum(r["audio_duration"] for r in results)
    throughput = total_audio / wall_time if wall_time > 0 else 0.0
    avg_rtf = mean(r["rtf"] for r in results) if results else 0.0
    effective_rtf = wall_time / total_audio if total_audio > 0 else 0.0

    results_sorted = sorted(results, key=lambda x: x["idx"])

    logs.append("\n--- Concurrent Summary ---")
    logs.append(f"Concurrent wall time: {wall_time:.2f}s")
    logs.append(f"Concurrent total audio: {total_audio:.2f}s")
    logs.append(f"Concurrent throughput: {throughput:.2f} audio-sec/wall-sec")
    logs.append(f"Concurrent effective/system RTF: {effective_rtf:.3f}")
    logs.append(f"Concurrent avg per-request RTF: {avg_rtf:.3f}")

    return wall_time, total_audio, results_sorted


def validate_outputs(seq_results, con_results, logs: list[str]):
    logs.append("\n==============================")
    logs.append("Output Duration Validation")
    logs.append("==============================")

    ok = True

    seq_by_idx = {r["idx"]: r for r in seq_results}
    con_by_idx = {r["idx"]: r for r in con_results}

    for idx in sorted(seq_by_idx):
        seq = seq_by_idx[idx]
        con = con_by_idx.get(idx)

        if con is None:
            logs.append(f"Text {idx}: missing concurrent output")
            ok = False
            continue

        seq_dur = seq["audio_duration"]
        con_dur = con["audio_duration"]

        ratio = con_dur / seq_dur if seq_dur > 0 else 0.0

        logs.append(
            f"Text {idx}: seq={seq_dur:.2f}s, "
            f"concurrent={con_dur:.2f}s, "
            f"duration_ratio={ratio:.2f}, "
            f"seq_chunks={seq['chunk_count']}, "
            f"con_chunks={con['chunk_count']}"
        )

        if con["chunk_count"] <= 1:
            ok = False

        if ratio < 0.60:
            ok = False

    if ok:
        logs.append("Validation: concurrent outputs look complete enough for batching comparison.")
    else:
        logs.append(
            "Validation: concurrent outputs still look incomplete. "
            "Do not claim end-to-end TTS batching yet."
        )

    return ok


async def benchmark_async(
    texts: list[str],
    voice: str,
    run_warmup_first: bool,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    min_tokens: int,
    repetition_penalty: float,
):
    model_obj = load_model_once()

    timestamp = int(time.time() * 1000)
    run_dir = BENCHMARK_OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs = []

    logs.append(f"Model: {MODEL_NAME}")
    logs.append(f"Voice: {voice}")
    logs.append(f"Run directory: {run_dir}")
    logs.append(f"Number of texts: {len(texts)}")
    logs.append(f"Concurrency: {concurrency}")
    logs.append(f"max_model_len: {MAX_MODEL_LEN}")
    logs.append(f"gpu_memory_utilization: {GPU_MEMORY_UTILIZATION}")
    logs.append(f"temperature: {temperature}")
    logs.append(f"top_p: {top_p}")
    logs.append(f"max_tokens: {max_tokens}")
    logs.append(f"min_tokens: {min_tokens}")
    logs.append(f"repetition_penalty: {repetition_penalty}")

    sampling_params = make_sampling_params(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        repetition_penalty=repetition_penalty,
    )

    if run_warmup_first:
        await warmup_async(
            model_obj=model_obj,
            voice=voice,
            sampling_params=sampling_params,
            logs=logs,
        )

    seq_wall, seq_audio, seq_results = await run_sequential_async(
        model_obj=model_obj,
        texts=texts,
        voice=voice,
        run_dir=run_dir,
        sampling_params=sampling_params,
        logs=logs,
    )

    con_wall, con_audio, con_results = await run_concurrent_async(
        model_obj=model_obj,
        texts=texts,
        voice=voice,
        run_dir=run_dir,
        sampling_params=sampling_params,
        concurrency=concurrency,
        logs=logs,
    )

    outputs_ok = validate_outputs(seq_results, con_results, logs)

    speedup = seq_wall / con_wall if con_wall > 0 else 0.0
    seq_throughput = seq_audio / seq_wall if seq_wall > 0 else 0.0
    con_throughput = con_audio / con_wall if con_wall > 0 else 0.0
    seq_effective_rtf = seq_wall / seq_audio if seq_audio > 0 else 0.0
    con_effective_rtf = con_wall / con_audio if con_audio > 0 else 0.0

    logs.append("\n==============================")
    logs.append("Final Comparison")
    logs.append("==============================")
    logs.append(f"Sequential wall time:       {seq_wall:.2f}s")
    logs.append(f"Concurrent wall time:       {con_wall:.2f}s")
    logs.append(f"Speedup:                    {speedup:.2f}x")
    logs.append(f"Sequential throughput:      {seq_throughput:.2f} audio-sec/wall-sec")
    logs.append(f"Concurrent throughput:      {con_throughput:.2f} audio-sec/wall-sec")
    logs.append(f"Sequential effective RTF:   {seq_effective_rtf:.3f}")
    logs.append(f"Concurrent effective RTF:   {con_effective_rtf:.3f}")
    logs.append(f"Output validation OK:       {outputs_ok}")

    if outputs_ok and speedup >= 1.5:
        verdict = "Concurrent vLLM inference gives meaningful speedup with valid audio durations."
    elif outputs_ok and speedup > 1.05:
        verdict = "Concurrent vLLM inference gives small speedup with valid audio durations."
    elif not outputs_ok:
        verdict = "Concurrent vLLM requests completed, but audio outputs look incomplete."
    else:
        verdict = "Little/no speedup."

    logs.append(f"Result: {verdict}")

    summary = (
        f"Seq: {seq_wall:.2f}s | "
        f"Concurrent: {con_wall:.2f}s | "
        f"Speedup: {speedup:.2f}x | "
        f"Concurrent effective RTF: {con_effective_rtf:.3f} | "
        f"Validation: {outputs_ok}"
    )

    row_labels = []

    seq_by_idx = {r["idx"]: r for r in seq_results}
    con_by_idx = {r["idx"]: r for r in con_results}

    for idx, text in enumerate(texts, start=1):
        seq = seq_by_idx.get(idx)
        con = con_by_idx.get(idx)

        seq_info = "missing"
        con_info = "missing"

        if seq:
            seq_info = (
                f"Sequential:\n"
                f"elapsed={seq['elapsed']:.2f}s\n"
                f"audio={seq['audio_duration']:.2f}s\n"
                f"RTF={seq['rtf']:.3f}\n"
                f"chunks={seq['chunk_count']}"
            )

        if con:
            con_info = (
                f"Concurrent:\n"
                f"elapsed={con['elapsed']:.2f}s\n"
                f"audio={con['audio_duration']:.2f}s\n"
                f"RTF={con['rtf']:.3f}\n"
                f"chunks={con['chunk_count']}"
            )

        row_labels.append(
            f"Text {idx}\n"
            f"{text}\n\n"
            f"{seq_info}\n\n"
            f"{con_info}"
        )

    row_labels = pad_items(row_labels, fill_value="")
    seq_audio_files = pad_items([r["path"] for r in seq_results])
    con_audio_files = pad_items([r["path"] for r in con_results])

    row_outputs = []
    for i in range(MAX_UI_AUDIOS):
        row_outputs.extend([row_labels[i], seq_audio_files[i], con_audio_files[i]])

    return summary, "\n".join(logs), *row_outputs


def benchmark_ui(
    texts_blob: str,
    voice: str,
    run_warmup_first: bool,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    min_tokens: int,
    repetition_penalty: float,
):
    texts = parse_texts(texts_blob)

    if not texts:
        raise gr.Error("Please enter at least one text line.")

    if len(texts) > MAX_UI_AUDIOS:
        raise gr.Error(f"Please use at most {MAX_UI_AUDIOS} texts for this UI.")

    if concurrency < 1:
        raise gr.Error("Concurrency must be at least 1.")

    if concurrency > len(texts):
        raise gr.Error("Concurrency cannot be greater than the number of text lines.")

    if min_tokens > max_tokens:
        raise gr.Error("min_tokens cannot be greater than max_tokens.")

    return asyncio.run(
        benchmark_async(
            texts=texts,
            voice=voice,
            run_warmup_first=run_warmup_first,
            concurrency=concurrency,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            repetition_penalty=repetition_penalty,
        )
    )


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Orpheus-TTS Sequential vs Concurrent vLLM Benchmark UI") as demo:
    gr.Markdown("# Orpheus-TTS Sequential vs Concurrent vLLM Benchmark UI")
    gr.Markdown(
        "This UI compares sequential generation against concurrent async-vLLM generation "
        "using one loaded Orpheus/vLLM engine. It uses token-id based decoding to avoid "
        "incomplete audio in concurrent mode."
    )

    with gr.Row():
        with gr.Column():
            texts_input = gr.Textbox(
                label="Texts, one per line",
                value=DEFAULT_TEXTS,
                lines=12,
            )

            voice_input = gr.Dropdown(
                label="Voice",
                choices=VOICE_OPTIONS,
                value=DEFAULT_VOICE,
            )

            run_warmup_checkbox = gr.Checkbox(
                label="Run warmup before benchmark",
                value=True,
            )

            concurrency_input = gr.Slider(
                label="Concurrent requests",
                minimum=1,
                maximum=MAX_UI_AUDIOS,
                value=4,
                step=1,
            )

            with gr.Accordion("Sampling parameters", open=False):
                temperature_input = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.5,
                    value=0.6,
                    step=0.05,
                )

                top_p_input = gr.Slider(
                    label="Top-p",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                )

                max_tokens_input = gr.Slider(
                    label="Max generated tokens",
                    minimum=100,
                    maximum=900,
                    value=900,
                    step=50,
                )

                min_tokens_input = gr.Slider(
                    label="Min generated tokens",
                    minimum=0,
                    maximum=300,
                    value=80,
                    step=10,
                )

                repetition_penalty_input = gr.Slider(
                    label="Repetition penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.3,
                    step=0.05,
                )

            run_button = gr.Button("Run Sequential vs Concurrent Benchmark")

        with gr.Column():
            summary_output = gr.Textbox(
                label="Summary",
                lines=4,
            )

            logs_output = gr.Textbox(
                label="Full Benchmark Logs",
                lines=24,
            )

    gr.Markdown("## One-to-One Audio Comparison")
    gr.Markdown("Left = text/details, middle = sequential audio, right = concurrent audio.")

    row_components = []

    for i in range(1, MAX_UI_AUDIOS + 1):
        with gr.Row():
            text_box = gr.Textbox(
                label=f"Text {i} Details",
                lines=9,
                interactive=False,
            )

            seq_audio = gr.Audio(
                label=f"Sequential Audio {i}",
                type="filepath",
            )

            con_audio = gr.Audio(
                label=f"Concurrent Audio {i}",
                type="filepath",
            )

        row_components.extend([text_box, seq_audio, con_audio])

    run_button.click(
        fn=benchmark_ui,
        inputs=[
            texts_input,
            voice_input,
            run_warmup_checkbox,
            concurrency_input,
            temperature_input,
            top_p_input,
            max_tokens_input,
            min_tokens_input,
            repetition_penalty_input,
        ],
        outputs=[
            summary_output,
            logs_output,
            *row_components,
        ],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    load_model_once()

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=6020,
        share=False,
    )
