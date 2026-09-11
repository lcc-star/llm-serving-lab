"""Open-loop scheduled arrivals at the scheduler boundary; no HTTP/tokenization timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

os.environ['MINISGL_DISABLE_OVERLAP_SCHEDULING'] = '1'
import torch
from minisgl.core import SamplingParams
from minisgl.llm.llm import LLM, RequestAllFinished
from minisgl.message import UserMsg
from metrics import summarize


class ReplayLLM(LLM):
    def reset_replay(self, rows, trace):
        self.rows = rows
        self.cursor = 0
        self.trace = trace
        self.events = []
        self.outputs = {r['uid']: [] for r in rows}
        self.done = set()
        # Prepare CPU input tensors outside timing; arrival delivery stays in the loop.
        self.inputs = {r['uid']: torch.tensor(r['input_ids'], dtype=torch.int32) for r in rows}
        self.start_time = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.start_time

    def offline_receive_msg(self, blocking=False):
        if blocking and self.cursor == len(self.rows):
            if len(self.done) != len(self.rows):
                raise RuntimeError('Scheduler idle with unfinished requests')
            raise RequestAllFinished()
        if blocking and self.cursor < len(self.rows):
            delay = self.rows[self.cursor]['arrival'] - self.now()
            if delay > 0:
                time.sleep(delay)
        msgs = []
        now = self.now()
        while self.cursor < len(self.rows) and self.rows[self.cursor]['arrival'] <= now:
            row = self.rows[self.cursor]
            self.cursor += 1
            if self.trace:
                self.events.append(dict(event='request', uid=row['uid'], arrival=row['arrival'],
                                        received=now, input_tokens=len(row['input_ids']),
                                        max_tokens=row['max_tokens']))
            msgs.append(UserMsg(uid=row['uid'], input_ids=self.inputs[row['uid']],
                                sampling_params=SamplingParams(temperature=0, ignore_eos=True,
                                                               max_tokens=row['max_tokens'])))
        return msgs

    def _schedule_next_batch(self):
        forward = super()._schedule_next_batch()
        if forward is not None and self.trace:
            batch = forward.batch
            free = len(self.cache_manager.free_slots)
            self.events.append(dict(event='batch', t=self.now(), phase=batch.phase,
                                    uids=[r.uid for r in batch.reqs], requests=batch.size,
                                    input_tokens=sum(r.extend_len for r in batch.reqs),
                                    pending=len(self.prefill_manager.pending_list),
                                    decode_running=len(self.decode_manager.running_reqs),
                                    kv_free_pages=free,
                                    kv_allocated_pages=self.cache_manager.num_pages-free))
        return forward

    def offline_send_result(self, reply):
        now = self.now() if self.trace else None
        for msg in reply:
            if msg.uid in self.done:
                raise RuntimeError('Duplicate output after completion')
            self.outputs[msg.uid].append(msg.next_token)
            if self.trace:
                reason = 'length' if msg.finished else None  # this harness forces ignore_eos
                self.events.append(dict(event='token', t=now, uid=msg.uid,
                                        token_id=msg.next_token, finished=msg.finished, reason=reason))
            if msg.finished:
                self.done.add(msg.uid)

    def replay(self, rows, trace):
        torch.cuda.synchronize()
        self.reset_replay(rows, trace)
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        torch.cuda.synchronize()
        elapsed = self.now()
        assert self.done == set(self.outputs)
        assert all(len(self.outputs[r['uid']]) == r['max_tokens'] for r in rows)
        self.cache_manager.check_integrity()
        assert self.table_manager.available_size == 8
        digest = hashlib.sha256(json.dumps(self.outputs, sort_keys=True).encode()).hexdigest()
        return dict(seconds=elapsed, output_tokens=sum(map(len, self.outputs.values())), sha256=digest)


def make_rows():
    rng = random.Random(42)
    # Delayed arrival and >prefill-budget input exercise queueing and chunk continuation.
    return [dict(uid=i, arrival=(0 if i < 4 else .15 + .03*(i-4)),
                 input_ids=[rng.randrange(100, 10000) for _ in range(128 if i < 4 else 512)],
                 max_tokens=64) for i in range(8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--workload', type=Path)
    parser.add_argument('--scenario', choices=['simultaneous', 'timed'], default='simultaneous')
    args = parser.parse_args()
    assert args.repeats > 0
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.workload.read_text()) if args.workload else make_rows()
    if not args.workload and args.scenario == 'simultaneous':
        for row in rows:
            row['arrival'] = 0.0
    assert rows and len({r['uid'] for r in rows}) == len(rows)
    assert all(0 <= r['arrival'] <= 30 and r['max_tokens'] > 0 and r['input_ids'] and
               len(r['input_ids']) + r['max_tokens'] <= 1024 for r in rows)
    assert rows == sorted(rows, key=lambda r: r['arrival'])
    (args.output/'workload.json').write_text(json.dumps(rows)+'\n')
    llm = ReplayLLM(args.model, cuda_graph_max_bs=8, cache_type='naive', num_page_override=8192,
                    page_size=1, max_running_req=8, max_seq_len_override=1024, max_extend_tokens=256)
    runs = []
    try:
        llm.replay(rows, False)  # warmup, excluded
        for repetition in range(args.repeats):
            # Alternate order to reduce bias from warming/drift.
            for trace in ((False, True) if repetition % 2 == 0 else (True, False)):
                record = dict(repetition=repetition, trace=trace, **llm.replay(rows, trace))
                if trace:
                    record['metrics'] = summarize(llm.events)
                    assert record['metrics']['completed'] == len(rows)
                    assert record['metrics']['output_tokens'] == record['output_tokens']
                    with (args.output/f'events_{repetition}.jsonl').open('w') as f:
                        for e in llm.events:
                            f.write(json.dumps(e)+'\n')
                (args.output/f'outputs_{repetition}_{int(trace)}.json').write_text(json.dumps(llm.outputs)+'\n')
                runs.append(record)
                print(json.dumps({k:v for k,v in record.items() if k != 'metrics'}), flush=True)
        outputs_match = len({r['sha256'] for r in runs}) == 1
        if not args.workload and args.scenario == 'simultaneous':
            assert outputs_match, 'Output mismatch in fixed-arrival control'
        medians = {str(trace): statistics.median(r['seconds'] for r in runs if r['trace']==trace)
                   for trace in (False, True)}
        result = dict(scenario=args.scenario, outputs_match=outputs_match, runs=runs, median_seconds=medians,
                      instrumentation_overhead_percent=(medians['True']/medians['False']-1)*100,
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      model=Path(args.model).name, graph=True, overlap=False, cache='naive',
                      note='Elapsed includes idle arrival waits. CPU-observed tokens; no HTTP. JSONL flushed outside timing.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        llm.shutdown()


if __name__ == '__main__':
    main()
