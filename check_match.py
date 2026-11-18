"""Why doesn't greedy speculative decoding always reproduce the baseline text?

Control: batch 1 and batch 4 use the same 16 prompts. If the *baseline* gives
different text at the two batch sizes, then mismatches come from fp16 numerics
and batching, not from speculative decoding.

    python check_match.py
"""

import json


def runs(name):
    with open(f"results/{name}.json") as f:
        data = json.load(f)
    return {(r["workload"], r["batch"]): r["texts"] for r in data["runs"]}


def first_diff(x, y):
    return next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))


def compare(a, b):
    diffs = [first_diff(x, y) for x, y in zip(a, b) if x != y]
    return len(a) - len(diffs), len(a), diffs


def main():
    base = runs("none")
    print("CONTROL: baseline at batch 1 vs baseline at batch 4 (same 16 prompts)")
    for w in ("edit", "open"):
        if (w, 1) in base and (w, 4) in base:
            same, n, diffs = compare(base[(w, 1)], base[(w, 4)])
            print(f"  {w}: {same}/{n} identical, first difference at char {diffs}")

    print("\nSPECULATIVE vs BASELINE at the same batch size")
    examples = []
    for name in ("ngram_k4", "draft_k4"):
        try:
            spec = runs(name)
        except (FileNotFoundError, KeyError):
            continue
        for (w, b), texts in sorted(spec.items()):
            same, n, diffs = compare(base[(w, b)], texts)
            print(f"  {name:9} {w:4} batch {b:2}: {same}/{n} identical"
                  + (f", first difference at char {diffs}" if diffs else ""))
            for x, y in zip(base[(w, b)], texts):
                if x != y and len(examples) < 2:
                    examples.append((name, w, b, x, y))

    for name, w, b, x, y in examples:
        i = first_diff(x, y)
        print(f"\nEXAMPLE {name}, {w}, batch {b}, split at char {i}:")
        print(f"  baseline: ...{x[max(0, i - 80):i + 60]!r}")
        print(f"  {name}: ...{y[max(0, i - 80):i + 60]!r}")


if __name__ == "__main__":
    main()
