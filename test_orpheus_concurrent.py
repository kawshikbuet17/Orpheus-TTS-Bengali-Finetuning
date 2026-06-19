import sys
sys.path.insert(0, "orpheus_tts_pypi")

import asyncio
import time
import wave
from pathlib import Path
from statistics import mean

from vllm import SamplingParams

from orpheus_tts import OrpheusModel
from orpheus_tts.decoder import tokens_decoder


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
VOICE = "tara"
SAMPLE_RATE = 24000

MAX_MODEL_LEN = 1024
GPU_MEMORY_UTILIZATION = 0.45

OUT_DIR = Path("benchmark_outputs/concurrent_vllm_token_ids")
OUT_DIR.mkdir(parents=True, exist_ok=True)


TEXTS = [
    (
        "Hello, this is the first Orpheus model level batching test. "
        "We are checking whether the first request can generate complete audio "
        "while other requests are active at the same time."
    ),
    (
        "This is the second request, submitted concurrently to the vLLM engine. "
        "The goal is to verify that asynchronous scheduling can process multiple "
        "speech token streams without cutting the audio short."
    ),
    (
        "The third request checks whether model level batching improves wall time. "
        "We will compare sequential generation with concurrent generation and also "
        "verify that the output duration is not suspiciously tiny."
    ),
    (
        "The fourth request measures total wall time and audio throughput. "
        "If batching works correctly, concurrent wall time should be lower while "
        "the generated audio should remain complete and playable."
    ),
]


# -----------------------------
# Sampling params
# -----------------------------
def make_sampling_params():
    return SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=900,
        min_tokens=80,
        stop_token_ids=[49158],
        repetition_penalty=1.3,
    )


# -----------------------------
# WAV helper
# -----------------------------
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


# -----------------------------
# vLLM token-id streaming
# -----------------------------
async def vllm_token_stream(model: OrpheusModel, text: str, request_id: str):
    """
    Direct async vLLM token stream for Orpheus.

    Why token_ids?
    - vLLM's output.text can be cumulative, chunked, or partial depending on streaming behavior.
    - Orpheus's decoder expects token-like strings containing custom audio-token patterns.
    - So we diff output.token_ids and decode each newly generated token ID individually.
    """
    prompt_string = model._format_prompt(text, VOICE)
    sampling_params = make_sampling_params()

    seen_token_count = 0
    event_count = 0
    yielded_token_count = 0
    fallback_text_count = 0

    async for result in model.engine.generate(
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
            # Fallback only. Less reliable for Orpheus custom audio tokens.
            piece = output.text
            if piece:
                fallback_text_count += 1
                yielded_token_count += 1
                yield piece
            continue

        # vLLM generally returns generated token_ids cumulatively.
        new_token_ids = token_ids[seen_token_count:]
        seen_token_count = len(token_ids)

        for token_id in new_token_ids:
            token_text = model.tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if token_text:
                yielded_token_count += 1
                yield token_text

    print(
        f"[{request_id}] events={event_count}, "
        f"generated_token_ids={seen_token_count}, "
        f"yielded_token_strings={yielded_token_count}, "
        f"fallback_text_chunks={fallback_text_count}"
    )


# -----------------------------
# Single generation
# -----------------------------
async def generate_one_async(model: OrpheusModel, text: str, idx: int, mode: str):
    out_path = OUT_DIR / f"{mode}_{idx}.wav"
    request_id = f"orpheus-{mode}-{idx}-{time.time_ns()}"

    start = time.monotonic()
    audio_chunks = []

    async for audio_chunk in tokens_decoder(
        vllm_token_stream(model, text, request_id)
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


# -----------------------------
# Sequential benchmark
# -----------------------------
async def run_sequential(model: OrpheusModel):
    print("\n==============================")
    print("Sequential async-vLLM run")
    print("==============================")

    start = time.monotonic()
    results = []

    for idx, text in enumerate(TEXTS, start=1):
        print(f"\nSequential item {idx}/{len(TEXTS)}")
        print(text)

        result = await generate_one_async(
            model=model,
            text=text,
            idx=idx,
            mode="seq",
        )

        results.append(result)

        print(
            f"seq {idx}: elapsed={result['elapsed']:.2f}s, "
            f"audio={result['audio_duration']:.2f}s, "
            f"RTF={result['rtf']:.3f}, "
            f"chunks={result['chunk_count']}, "
            f"path={result['path']}"
        )

    wall_time = time.monotonic() - start
    total_audio = sum(r["audio_duration"] for r in results)
    throughput = total_audio / wall_time if wall_time > 0 else 0.0

    print("\n--- Sequential Summary ---")
    print(f"Sequential wall time: {wall_time:.2f}s")
    print(f"Sequential total audio: {total_audio:.2f}s")
    print(f"Sequential throughput: {throughput:.2f} audio-sec/wall-sec")
    print(f"Sequential avg RTF: {mean(r['rtf'] for r in results):.3f}")

    return wall_time, total_audio, results


# -----------------------------
# Concurrent benchmark
# -----------------------------
async def run_concurrent(model: OrpheusModel, concurrency: int):
    print("\n==============================")
    print(f"Concurrent async-vLLM run: concurrency={concurrency}")
    print("==============================")

    texts = TEXTS[:concurrency]

    start = time.monotonic()

    tasks = [
        asyncio.create_task(
            generate_one_async(
                model=model,
                text=text,
                idx=idx,
                mode="concurrent",
            )
        )
        for idx, text in enumerate(texts, start=1)
    ]

    results = []

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)

        print(
            f"concurrent {result['idx']}: elapsed={result['elapsed']:.2f}s, "
            f"audio={result['audio_duration']:.2f}s, "
            f"RTF={result['rtf']:.3f}, "
            f"chunks={result['chunk_count']}, "
            f"path={result['path']}"
        )

    wall_time = time.monotonic() - start
    total_audio = sum(r["audio_duration"] for r in results)
    throughput = total_audio / wall_time if wall_time > 0 else 0.0

    results_sorted = sorted(results, key=lambda x: x["idx"])

    print("\n--- Concurrent Summary ---")
    print(f"Concurrent wall time: {wall_time:.2f}s")
    print(f"Concurrent total audio: {total_audio:.2f}s")
    print(f"Concurrent throughput: {throughput:.2f} audio-sec/wall-sec")
    print(f"Concurrent avg RTF: {mean(r['rtf'] for r in results):.3f}")

    return wall_time, total_audio, results_sorted


# -----------------------------
# Warmup
# -----------------------------
async def warmup(model: OrpheusModel):
    print("\nWarmup...")

    warmup_text = (
        "This is a short warmup request before measuring Orpheus concurrent "
        "model level batching."
    )

    chunks = []

    async for audio_chunk in tokens_decoder(
        vllm_token_stream(model, warmup_text, f"warmup-{time.time_ns()}")
    ):
        chunks.append(audio_chunk)

    print(f"Warmup done. Warmup chunks={len(chunks)}")


# -----------------------------
# Validation
# -----------------------------
def validate_outputs(seq_results, con_results):
    print("\n==============================")
    print("Output Duration Validation")
    print("==============================")

    ok = True

    seq_by_idx = {r["idx"]: r for r in seq_results}
    con_by_idx = {r["idx"]: r for r in con_results}

    for idx in sorted(seq_by_idx):
        seq = seq_by_idx[idx]
        con = con_by_idx.get(idx)

        if con is None:
            print(f"Text {idx}: missing concurrent output")
            ok = False
            continue

        seq_dur = seq["audio_duration"]
        con_dur = con["audio_duration"]

        ratio = con_dur / seq_dur if seq_dur > 0 else 0.0

        print(
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
        print("Validation: concurrent outputs look complete enough for batching comparison.")
    else:
        print(
            "Validation: concurrent outputs still look incomplete. "
            "Do not claim end-to-end TTS batching yet."
        )

    return ok


# -----------------------------
# Benchmark orchestration
# -----------------------------
async def benchmark_async(model: OrpheusModel):
    await warmup(model)

    seq_wall, seq_audio, seq_results = await run_sequential(model)

    con_wall, con_audio, con_results = await run_concurrent(
        model=model,
        concurrency=4,
    )

    outputs_ok = validate_outputs(seq_results, con_results)

    speedup = seq_wall / con_wall if con_wall > 0 else 0.0
    seq_throughput = seq_audio / seq_wall if seq_wall > 0 else 0.0
    con_throughput = con_audio / con_wall if con_wall > 0 else 0.0

    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    print(f"Sequential wall time:  {seq_wall:.2f}s")
    print(f"Concurrent wall time:  {con_wall:.2f}s")
    print(f"Speedup:               {speedup:.2f}x")
    print(f"Sequential throughput: {seq_throughput:.2f} audio-sec/wall-sec")
    print(f"Concurrent throughput: {con_throughput:.2f} audio-sec/wall-sec")
    print(f"Output validation OK:  {outputs_ok}")

    if outputs_ok and speedup >= 1.5:
        verdict = "Concurrent vLLM inference gives meaningful speedup with valid audio durations."
    elif outputs_ok and speedup > 1.05:
        verdict = "Concurrent vLLM inference gives small speedup with valid audio durations."
    elif not outputs_ok:
        verdict = "Concurrent vLLM requests completed, but audio outputs look incomplete."
    else:
        verdict = "Little/no speedup."

    print(f"Result: {verdict}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading Orpheus model once...")
    print(f"Model: {MODEL_NAME}")
    print(f"max_model_len={MAX_MODEL_LEN}")
    print(f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")

    model = OrpheusModel(
        model_name=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    asyncio.run(benchmark_async(model))


if __name__ == "__main__":
    main()
