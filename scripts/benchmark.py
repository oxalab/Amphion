"""Amphion benchmark script -- the untrusted executor boundary.

Standalone on purpose: ZERO Amphion imports. It ingests config, prints ONE
JSON blob to stdout, exits 0 on success / 1 on failure. RunBenchmark
subprocesses this and parses the JSON. Keep stdout pure -- anything else
printed to stdout breaks the parser.

Usage:
    uv run python scripts/benchmark.py --model_path Qwen/Qwen3-0.6B --dtype float16
"""

import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = (
    "Large language model inference performance depends on memory bandwidth, "
    "kernel efficiency, and batching strategy. Explain the key tradeoffs."
)


def parse_args():
    p = argparse.ArgumentParser(description="Measure TTFT / decode speed / peak VRAM.")
    _ = p.add_argument("--model_path", required=True,
                   help="local cache path OR huggingface repo id")
    _ = p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    _ = p.add_argument("--n_tokens", type=int, default=64)
    _ = p.add_argument("--batch_size", type=int, default=1)
    _ = p.add_argument("--prompt", default=DEFAULT_PROMPT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    provided_dtype = args.dtype
    dtype: torch.dtype
    if isinstance(provided_dtype, str):
        # Using provided_dtype here ensures strict type parsing
        dtype = torch.float16 if provided_dtype == "float16" else torch.float32
    else:
        # If it is already a torch.dtype object or another non-string fallback
        dtype = provided_dtype if isinstance(provided_dtype, torch.dtype) else torch.float32
    # --- load: dtype AT LOAD TIME. Converting later would leave an fp32 copy
    # resident in VRAM and corrupt the memory measurements. ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        assert tokenizer is not None
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
        torch.nn.Module.to(model, "cuda")
        model.eval()
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during load: {e}", file=sys.stderr)
        return 1

    # --- warmup: first CUDA calls JIT-compile kernels; timing them pollutes results ---
    try:
        with torch.no_grad():
            warm = tokenizer([args.prompt] * args.batch_size, return_tensors="pt").to("cuda")
            model.generate(warm.input_ids, max_new_tokens=2, do_sample=False)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during warmup: {e}", file=sys.stderr)
        return 1

    # peak = GENERATION only (model weights + warmup excluded by this reset)
    torch.cuda.reset_peak_memory_stats()

    try:
        with torch.no_grad():
            # --- TTFT: prompt processing alone, before any generation ---
            input_ids = tokenizer([args.prompt] * args.batch_size, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()          # sync pair: clean start...
            t0 = time.perf_counter()
            model(input_ids.input_ids)        # prefill only
            torch.cuda.synchronize()          # ...and wait for the GPU to actually finish
            ttft_ms = (time.perf_counter() - t0) * 1000

            # --- decode: greedy, full re-process each step. This is the BASELINE --
            # deliberately no KV-cache tricks; the use_cache comparison is a
            # rung-2 experiment on this same script. ---
            seq = input_ids.input_ids
            step_times = []
            for _ in range(args.n_tokens):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = model(seq)
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, next_tok], dim=-1)
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t)
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during generation: {e}", file=sys.stderr)
        return 1

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # first decode step carries residual prefill/JIT effects -- exclude from the rate
    decode_times = step_times[1:]
    tokens_per_sec = len(decode_times) / sum(decode_times) if decode_times else 0.0

    result = {
        "model": args.model_path,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "n_tokens": args.n_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "total_time_ms": round(ttft_ms + sum(step_times) * 1000, 2),
    }
    print(json.dumps(result))               # the ONLY stdout line
    return 0


if __name__ == "__main__":
    sys.exit(main())
