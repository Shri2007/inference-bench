"""Turn results/*.json into charts and a markdown table.

    python report.py

Writes results/throughput.png, results/speedup.png and results/summary.md.
It also checks that every speculative config produced the same text as the
baseline, since greedy speculative decoding is supposed to be lossless.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"none": "#4C4C4C", "ngram": "#1F77B4", "draft": "#D95F02"}


def load():
    configs = {}
    for path in sorted(glob.glob("results/*.json")):
        with open(path) as f:
            data = json.load(f)
        configs[data["meta"]["name"]] = data
    return configs


def index(data):
    return {(r["workload"], r["batch"]): r for r in data["runs"]}


def match_rate(base_run, run):
    pairs = list(zip(base_run["texts"], run["texts"]))
    if not pairs:
        return None
    return sum(a == b for a, b in pairs) / len(pairs)


def main():
    configs = load()
    if "none" not in configs:
        raise SystemExit("results/none.json is missing. Run `python bench.py --spec none` first.")
    base = index(configs["none"])
    workloads = sorted({w for w, _ in base})
    batches = sorted({b for _, b in base})

    failed = {n: d["meta"]["error"] for n, d in configs.items() if d["meta"].get("error")}
    ok = {n: d for n, d in configs.items() if not d["meta"].get("error")}
    ok = dict(sorted(ok.items(), key=lambda kv: (kv[0] != "none", kv[0])))

    # Chart 1: throughput vs batch size, one panel per workload.
    fig, axes = plt.subplots(1, len(workloads), figsize=(5.5 * len(workloads), 4), sharey=True)
    axes = axes if len(workloads) > 1 else [axes]
    for ax, w in zip(axes, workloads):
        for name, data in ok.items():
            idx = index(data)
            xs = [b for b in batches if (w, b) in idx]
            ys = [idx[(w, b)]["output_tok_per_s"] for b in xs]
            ax.plot(xs, ys, marker="o", label=name, color=COLORS.get(data["meta"]["spec"]))
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches, [str(b) for b in batches])
        ax.set_title(f"{w} workload")
        ax.set_xlabel("concurrent requests (batch size)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("output tokens / second")
    axes[-1].legend()
    gpu = configs["none"]["meta"]["gpu"]
    fig.suptitle(f"{configs['none']['meta']['model']} on {gpu}")
    fig.tight_layout()
    fig.savefig("results/throughput.png", dpi=150)

    # Chart 2: speedup over baseline at each batch size.
    spec_names = [n for n in ok if n != "none"]
    if spec_names:
        fig, axes = plt.subplots(1, len(workloads), figsize=(5.5 * len(workloads), 4), sharey=True)
        axes = axes if len(workloads) > 1 else [axes]
        width = 0.8 / len(spec_names)
        for ax, w in zip(axes, workloads):
            for i, name in enumerate(spec_names):
                idx = index(ok[name])
                xs = [j + i * width for j, b in enumerate(batches) if (w, b) in idx]
                ys = [idx[(w, b)]["output_tok_per_s"] / base[(w, b)]["output_tok_per_s"]
                      for b in batches if (w, b) in idx]
                ax.bar(xs, ys, width, label=name, color=COLORS.get(ok[name]["meta"]["spec"]))
            ax.axhline(1.0, color="black", linewidth=1)
            ax.set_xticks([j + width * (len(spec_names) - 1) / 2 for j in range(len(batches))],
                          [str(b) for b in batches])
            ax.set_title(f"{w} workload")
            ax.set_xlabel("concurrent requests (batch size)")
        axes[0].set_ylabel("speedup vs. no speculation")
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig("results/speedup.png", dpi=150)

    # Markdown summary.
    lines = [f"Model: `{configs['none']['meta']['model']}` on **{gpu}**, "
             f"vLLM {configs['none']['meta']['vllm']}, greedy, max {configs['none']['meta']['max_tokens']} tokens.", ""]
    lines.append("| workload | batch | config | tok/s | speedup | acceptance | same text as baseline |")
    lines.append("|---|---|---|---|---|---|---|")
    for w in workloads:
        for b in batches:
            if (w, b) not in base:
                continue
            for name, data in ok.items():
                r = index(data).get((w, b))
                if not r:
                    continue
                speed = r["output_tok_per_s"] / base[(w, b)]["output_tok_per_s"]
                acc = f"{r['acceptance_rate']:.0%}" if "acceptance_rate" in r else "-"
                same = "-" if name == "none" else f"{match_rate(base[(w, b)], r):.0%}"
                lines.append(f"| {w} | {b} | {name} | {r['output_tok_per_s']:.0f} | {speed:.2f}x | {acc} | {same} |")
    for name, err in failed.items():
        lines.append(f"\n`{name}` did not start: {err.splitlines()[0]}")
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote results/throughput.png, results/speedup.png, results/summary.md")


if __name__ == "__main__":
    main()
