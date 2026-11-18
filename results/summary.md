Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` on **Tesla T4**, vLLM 0.34.0, greedy, max 512 tokens.

| workload | batch | config | tok/s | speedup | acceptance | same text as baseline |
|---|---|---|---|---|---|---|
| edit | 1 | none | 63 | 1.00x | - | - |
| edit | 1 | draft_k4 | 101 | 1.61x | 95% | 100% |
| edit | 1 | ngram_k4 | 187 | 2.99x | 57% | 100% |
| edit | 4 | none | 144 | 1.00x | - | - |
| edit | 4 | draft_k4 | 242 | 1.68x | 95% | 100% |
| edit | 4 | ngram_k4 | 403 | 2.80x | 57% | 100% |
| edit | 16 | none | 415 | 1.00x | - | - |
| edit | 16 | draft_k4 | 560 | 1.35x | 95% | 100% |
| edit | 16 | ngram_k4 | 799 | 1.93x | 54% | 100% |
| edit | 64 | none | 888 | 1.00x | - | - |
| edit | 64 | draft_k4 | 937 | 1.05x | 94% | 100% |
| edit | 64 | ngram_k4 | 1248 | 1.41x | 53% | 100% |
| open | 1 | none | 61 | 1.00x | - | - |
| open | 1 | draft_k4 | 76 | 1.25x | 53% | 81% |
| open | 1 | ngram_k4 | 69 | 1.13x | 2% | 81% |
| open | 4 | none | 158 | 1.00x | - | - |
| open | 4 | draft_k4 | 184 | 1.16x | 54% | 94% |
| open | 4 | ngram_k4 | 155 | 0.98x | 3% | 94% |
| open | 16 | none | 486 | 1.00x | - | - |
| open | 16 | draft_k4 | 474 | 0.98x | 54% | 91% |
| open | 16 | ngram_k4 | 383 | 0.79x | 3% | 91% |
| open | 64 | none | 1249 | 1.00x | - | - |
| open | 64 | draft_k4 | 990 | 0.79x | 55% | 92% |
| open | 64 | ngram_k4 | 710 | 0.57x | 3% | 80% |
