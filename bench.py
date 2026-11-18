"""Run one vLLM engine config across workloads and batch sizes.

Each config runs in its own process because vLLM does not reliably free GPU
memory when an engine is deleted. Results go to results/<name>.json.

    python bench.py --spec none
    python bench.py --spec ngram --k 4
    python bench.py --spec draft --k 4
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback

import prompts

BATCH_SIZES = [1, 4, 16, 64]


def n_prompts_for(batch):
    # Batch 1 is slow, so it gets fewer prompts. Larger batches need enough
    # prompts to fill several batches.
    return max(16, batch * 2) if batch < 64 else 64


def spec_config(args):
    if args.spec == "none":
        return None
    if args.spec == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": args.k,
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
        }
    if args.spec == "draft":
        return {"model": args.draft_model, "num_speculative_tokens": args.k}
    raise ValueError(args.spec)


def read_spec_counters(llm):
    """Return (drafted, accepted) token counts, or None if this vLLM can't report them."""
    try:
        drafted = accepted = 0
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_draft_tokens":
                drafted += m.value
            elif m.name == "vllm:spec_decode_num_accepted_tokens":
                accepted += m.value
        return drafted, accepted
    except Exception:
        return None


def run(llm, sampling, convs, batch):
    """Generate for all convs, `batch` requests at a time. Returns stats + texts."""
    out_tokens = 0
    texts = []
    t0 = time.perf_counter()
    for i in range(0, len(convs), batch):
        outs = llm.chat(convs[i:i + batch], sampling, use_tqdm=False)
        for o in outs:
            out_tokens += len(o.outputs[0].token_ids)
            texts.append(o.outputs[0].text)
    elapsed = time.perf_counter() - t0
    n_batches = -(-len(convs) // batch)
    return {
        "n_prompts": len(convs),
        "output_tokens": out_tokens,
        "seconds": round(elapsed, 3),
        "output_tok_per_s": round(out_tokens / elapsed, 2),
        "seconds_per_batch": round(elapsed / n_batches, 3),
    }, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--draft-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--spec", choices=["none", "ngram", "draft"], default="none")
    p.add_argument("--k", type=int, default=4, help="speculative tokens per step")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--batches", default=",".join(map(str, BATCH_SIZES)))
    p.add_argument("--workloads", default="edit,open")
    p.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    p.add_argument("--name", default=None, help="results file name (default: spec name)")
    args = p.parse_args()

    name = args.name or (args.spec if args.spec == "none" else f"{args.spec}_k{args.k}")
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{name}.json")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    meta = {
        "name": name,
        "spec": args.spec,
        "k": args.k if args.spec != "none" else 0,
        "model": args.model,
        "draft_model": args.draft_model if args.spec == "draft" else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vllm": vllm.__version__,
        "python": platform.python_version(),
        "max_tokens": args.max_tokens,
    }
    print(json.dumps(meta, indent=2), flush=True)

    # T4 and other pre-Ampere GPUs have no bf16, so use fp16 everywhere for a fair comparison.
    try:
        llm = LLM(
            model=args.model,
            dtype="half",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            seed=0,
            disable_log_stats=False,
            enforce_eager=args.enforce_eager,
            speculative_config=spec_config(args),
        )
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc()
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "runs": []}, f, indent=2)
        print(f"engine failed to start, wrote {out_path}", file=sys.stderr)
        sys.exit(1)

    # Greedy decoding: speculative decoding should then give the same text as baseline.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warm up so CUDA graph capture and compilation don't count against batch 1.
    llm.chat(prompts.build("open", 2), SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)

    runs = []
    for workload in args.workloads.split(","):
        for batch in [int(b) for b in args.batches.split(",")]:
            convs = prompts.build(workload, n_prompts_for(batch))
            before = read_spec_counters(llm)
            stats, texts = run(llm, sampling, convs, batch)
            after = read_spec_counters(llm)
            if before and after and after[0] > before[0]:
                drafted, accepted = after[0] - before[0], after[1] - before[1]
                stats["draft_tokens"] = int(drafted)
                stats["accepted_tokens"] = int(accepted)
                stats["acceptance_rate"] = round(accepted / drafted, 4)
            stats.update(workload=workload, batch=batch, texts=texts)
            runs.append(stats)
            summary = {k: v for k, v in stats.items() if k != "texts"}
            print(json.dumps(summary), flush=True)
            # Save after every run so a Colab disconnect doesn't lose everything.
            with open(out_path, "w") as f:
                json.dump({"meta": meta, "runs": runs}, f, indent=2)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
