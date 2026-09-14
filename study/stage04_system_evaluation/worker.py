"""One engine per KV capacity; private resumable runs, public aggregate metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage01_measurement'))
from replay import ReplayLLM, torch
from minisgl.scheduler.policy import BatchPolicy
from analysis import analyze
from workloads import make_workload, workload_hash


class EvaluationLLM(ReplayLLM):
    def reset_replay(self, rows, trace):
        super().reset_replay(rows, trace)
        self.batch_policy = BatchPolicy(self.policy, decode_wait_ms=50, prefill_wait_ms=200)

    def offline_receive_msg(self, blocking=False):
        if self.now() > self.deadline_s:
            raise TimeoutError('Replay exceeded its declared time limit')
        return super().offline_receive_msg(blocking)

    def _record_scheduling_decision(self, decision):
        if self.trace:
            self.events.append(dict(event='decision', t=self.now(), **decision))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--pools', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pages', type=int, required=True)
    args = parser.parse_args()
    pools = json.loads(args.pools.read_text())
    jobs = json.loads(args.jobs.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    # Resume completed immutable jobs only; no skip after a failed partial replay.
    pending = []
    for job in jobs:
        path = args.output / (job['name'] + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['job'] == job and saved['status'] == 'ok'
        else:
            pending.append(job)
    if not pending:
        return
    llm = EvaluationLLM(args.model, cuda_graph_max_bs=8, cache_type='naive',
                        num_page_override=args.pages, page_size=1, max_running_req=8,
                        max_seq_len_override=8192, max_extend_tokens=4096)
    warmed = set()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    try:
        for job in pending:
            assert job['pages'] == args.pages
            rows = make_workload(pools[job['split']], job['scenario'], job['count'], job['rate'], job['seed'])
            assert all(len(r['input_ids'])+r['max_tokens'] <= min(args.pages,8192) for r in rows)
            llm.policy = job['policy']
            llm.prefill_budget = job['budget']
            llm.deadline_s = max(120, rows[-1]['arrival']*2 + 120)
            key = (job['scenario'], job['level'], job['policy'], job['budget'])
            if key not in warmed:
                # Warm model shapes with the same mixture, excluding waits and timed output.
                warm = [dict(r, arrival=0.0, max_tokens=8) for r in rows[:8]]
                llm.replay(warm, False)
                warmed.add(key)
            record = dict(job=job, status='running', source_commit=commit,
                          workload_sha256=workload_hash(rows), gpu=torch.cuda.get_device_name(),
                          torch=torch.__version__, model=Path(args.model).name)
            try:
                result = llm.replay(rows, True)
                assert all(not v for v in llm.batch_policy.wait_since.values())
                assert all(e['input_tokens'] <= job['budget'] for e in llm.events
                           if e['event']=='batch' and e['phase']=='prefill')
                metrics = analyze(llm.events, rows)
                assert metrics['completed'] == len(rows)
                record.update(status='ok', **result, metrics=metrics)
            except Exception as error:
                record.update(status='failed', error_type=type(error).__name__)
                (args.output/(job['name']+'.failed.json')).write_text(json.dumps(record,indent=2)+'\n')
                raise
            # These files are private: never publish raw token IDs or request traces.
            (args.output/(job['name']+'.jsonl')).write_text(''.join(json.dumps(e)+'\n' for e in llm.events))
            (args.output/(job['name']+'_outputs.json')).write_text(json.dumps(llm.outputs)+'\n')
            (args.output/(job['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(dict(name=job['name'],seconds=round(result['seconds'],3),
                                  max_pending=metrics['max_pending'],completed=metrics['completed'])),flush=True)
    finally:
        llm.shutdown()


if __name__ == '__main__':
    main()
