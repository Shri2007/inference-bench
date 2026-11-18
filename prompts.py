"""Builds the two benchmark workloads.

edit: add type hints to a real Python stdlib function. The answer mostly
      copies the prompt, like a coding agent applying a small edit.
open: write a short explanation. The answer shares little text with the prompt.

Everything comes from the local Python install, so no dataset download is needed.
"""

import importlib
import inspect
import random

STDLIB_MODULES = [
    "argparse", "base64", "bisect", "calendar", "colorsys", "configparser",
    "csv", "difflib", "email.utils", "fnmatch", "fractions", "functools",
    "glob", "gzip", "heapq", "ipaddress", "json.decoder", "json.encoder",
    "ntpath", "posixpath", "pprint", "shutil", "statistics", "string",
    "tarfile", "textwrap", "tokenize", "urllib.parse", "zipfile",
]

EDIT_INSTRUCTION = (
    "Add Python type hints to the parameters and return value of this function. "
    "Do not change anything else. Reply with the complete updated function in a "
    "single ```python code block and nothing else.\n\n```python\n{code}```"
)

OPEN_TOPICS = [
    "how a hash map handles collisions", "what a database index is",
    "how TCP retransmission works", "what garbage collection does",
    "how HTTPS keeps traffic private", "what a race condition is",
    "how a CPU cache works", "what eventual consistency means",
    "how a load balancer picks a server", "what a memory leak is",
    "how DNS resolves a domain name", "what a mutex is for",
    "how git stores history", "what backpressure means in a queue",
    "how a B-tree stays balanced", "what a deadlock is",
    "how virtual memory works", "what idempotency means for an API",
    "how a bloom filter works", "what a container image contains",
    "how rate limiting works", "what a write-ahead log is",
    "how consistent hashing works", "what a CDN does",
    "how an LLM tokenizer works", "what the KV cache in a transformer stores",
    "how gradient descent works", "what overfitting is",
    "how a compiler turns code into machine code", "what a thread pool is",
    "how public key cryptography works", "what a message queue is for",
]
OPEN_INSTRUCTION = "Write a clear 250-word explanation of {topic} for a new software engineer."


def _stdlib_functions(min_lines=10, max_lines=45):
    found = []
    for name in STDLIB_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for fname, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__ or fname.startswith("__"):
                continue
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            n = src.count("\n")
            if min_lines <= n <= max_lines and "->" not in src.split("\n", 1)[0]:
                found.append((f"{name}.{fname}", src))
    found.sort()
    return found


def build(workload, n, seed=0):
    """Return n chat conversations (lists of message dicts) for a workload."""
    rng = random.Random(seed)
    if workload == "edit":
        funcs = _stdlib_functions()
        if len(funcs) < n:
            raise ValueError(f"only found {len(funcs)} stdlib functions, need {n}")
        picked = rng.sample(funcs, n)
        texts = [EDIT_INSTRUCTION.format(code=src) for _, src in picked]
    elif workload == "open":
        topics = [OPEN_TOPICS[i % len(OPEN_TOPICS)] for i in range(n)]
        rng.shuffle(topics)
        texts = [OPEN_INSTRUCTION.format(topic=t) for t in topics]
    else:
        raise ValueError(f"unknown workload {workload!r}")
    return [[{"role": "user", "content": t}] for t in texts]


if __name__ == "__main__":
    for w in ("edit", "open"):
        convs = build(w, 64)
        print(f"{w}: {len(convs)} prompts")
        print(convs[0][0]["content"][:400], "\n---")
    print("stdlib functions available:", len(_stdlib_functions()))
