# Stage 1 measurement prototype — 2026-09-11

Implementation tested at commit `528ba47` (recorded in each summary). Engine scheduling policy is unchanged. Python modules were loaded from this new repository using explicit `PYTHONPATH`, with dependencies reused from the existing virtual environment.

## Delivered

- Fixed intended-arrival replay at the scheduler boundary, plus observed ingress lag.
- CPU request, batch and per-token event recording; buffered JSONL export.
- TTFT, pooled ITL, E2E, arrival-to-first-schedule, throughput and completion metrics.
- Per-batch phase, token/request count, pending/running count and KV page accounting.
- Independent analysis and standalone SVG timeline.
- Unit tests and real GPU completion/output/resource checks.

## Experiment

A800 single GPU; Llama-3.1-8B-Instruct BF16; CUDA Graph enabled; overlap disabled; naive cache; 8192 one-token KV pages; 256 prefill token budget. Eight requests: four 128-token inputs and four 512-token inputs, all generating exactly 64 tokens with ignore_eos. No HTTP or tokenizer latency is measured. Each scenario has one excluded warmup and five traced/five untraced trials in alternating order.

| Control (all arrivals at zero) | Median replay seconds |
|---|---:|
| Event recording off | 0.999251885 |
| Event recording on | 1.000010542 |

Incremental event-recording overhead: **0.0759%** in this small experiment. This is below a level where we claim a precise general overhead estimate. Both modes retain token collection and correctness checks; file serialization is excluded. All ten runs produced 512 tokens and identical output hashes. Request-table recovery and cache integrity checks passed.

## Timed-arrival diagnostics

Four requests arrive at zero, followed by four at 0.15/0.18/0.21/0.24 seconds. All ten runs completed all eight requests and 512 tokens with resource checks passing. Outputs were not identical across all runs; raw output IDs and first-divergence locations are preserved in `evidence/timed/output_differences.json`. Variable batch composition is a hypothesis, not an established explanation. This remains a correctness investigation before treating timed replay as an output-equivalence benchmark.

Representative recorded run 0 (not aggregate across repetitions): TTFT P50 47.35 ms; ITL P50 11.83 ms; maximum observed ITL 216.57 ms; maximum ingress lag 19.10 ms. Tail metrics are descriptive only for this small workload. The timeline marks CPU events, not GPU kernel durations. These numbers validate visibility into pauses, not a scheduling optimization.

## Validation and remaining scope

Four standard-library unit tests passed: known timeline, empty/incomplete results, reversed timestamps, single-token/no-ITL and duplicate completion. JSONL reanalysis completed and SVGs parsed as XML. GPU memory was released after experiments.

This prototype is single-GPU, normal scheduling, fixed-length synthetic token replay. It does not yet measure HTTP client latency, GPU event duration, cancellation or natural EOS. Timed-arrival output differences remain unresolved. P99 claims require larger samples. Those boundaries must be addressed explicitly in subsequent online/correctness work; no stage-2 scheduling change has been made.
