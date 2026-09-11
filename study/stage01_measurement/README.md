# Stage 1: scheduler-boundary measurement

This harness subclasses the existing Scheduler through LLM without changing scheduling policy. It supports single-GPU normal scheduling only. CUDA Graph is enabled, overlap disabled, prefix cache naive. EOS is ignored and output lengths are fixed for performance validation. Network, tokenizer, cancellation and natural-EOS experiments are not implemented here.

## Run

Activate an environment with the project dependencies, then run from repository root:

```bash
export PYTHONPATH="$PWD/python"
python -m unittest discover -s tests/stage01_measurement -v
CUDA_VISIBLE_DEVICES=0 python study/stage01_measurement/replay.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --output study/stage01_measurement/runs/example
python study/stage01_measurement/analyze.py \
  study/stage01_measurement/runs/example/events_0.jsonl \
  --output study/stage01_measurement/runs/example/analysis
```

Optional `--workload file.json` accepts an arrival-sorted JSON array of `{uid, arrival, input_ids, max_tokens}`. IDs must be unique, arrivals are relative seconds (0..30), input+output must fit 1024 tokens. Inputs must be valid IDs for the selected model. The harness delivers all due requests at each scheduler receive boundary; it never waits for a response before advancing the arrival schedule. Late delivery is recorded as ingress lag. This is a scheduler-boundary replay, not a separate network load generator. CPU tensors are prepared before timing.

## Time definitions

- arrival: intended arrival from the fixed workload, relative to replay start.
- received: when the scheduler observes the request. `received-arrival` is ingress lag.
- batch.t: after batch preparation, before forward submission; first such event defines first scheduling here (includes preparation overhead).
- token.t: CPU time at result delivery after the existing copy-completion wait and result handling. Same-batch tokens share a timestamp. This is neither precise GPU completion time nor client receipt time.
- TTFT: first token.t minus intended arrival; includes ingress lag.
- ITL: adjacent CPU-observed token times within each request; percentiles pool these gaps across requests.
- E2E: last token.t minus intended arrival for completed requests.
- arrival-to-first-schedule: first batch.t minus arrival.
- token throughput: observed output count / (last token time - earliest intended arrival).
- overhead: traced/untraced median full-replay elapsed ratio minus one; elapsed includes intentional arrival gaps and final synchronization.

Batch records include phase, true request/token counts, pending/running sizes and allocated/free KV pages. Allocated pages include any retained cache; they are not a CUDA memory utilization metric. Records are buffered in RAM and serialized after the timed interval. This prototype is intended for bounded traces; memory grows with event count. No per-round global CUDA synchronization is added.

## Validation and limits

The default simultaneous-arrival control isolates instrumentation overhead. Use `--scenario timed` for delayed-arrival validation. Timed-output differences are retained for investigation, not accepted as proof of numerical correctness. One warmup is excluded. Five traced and five untraced trials alternate order. Both modes collect outputs and perform completion checks, so overhead measures extra event recording, not the full harness overhead versus a server. Every request must return its specified length; output hashes must match in the simultaneous-arrival control; timed-arrival hashes are reported separately because scheduling can change batch composition; cache integrity and table-slot recovery are checked after each trial. Unit tests cover known metrics, absent results, invalid ordering, single-token output and duplicate completion.

P99 is only descriptive for the small smoke workload, not a statistically reliable tail claim. This stage does not prove scheduling improvements or predict online client latency. JSONL preserves raw records for independent analysis. Long-prefill interference experiments and production trace replay are subsequent stages.

## Output divergence follow-up

See [中文调查报告](DIVERGENCE_REPORT.md) for fixed-step interventions reproducing BF16 candidate ties. Run `python study/stage01_measurement/validate_divergence.py` to validate the saved controls on CPU. This explains the identified uid=1 token-60 divergence, not arbitrary future mismatches.
