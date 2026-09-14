"""Replay recorded batches and inspect logits. Diagnostic only, never performance timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage01_measurement'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage04_system_evaluation'))
from replay import ReplayLLM, torch, UserMsg, SamplingParams, RequestAllFinished
from workloads import make_workload, workload_hash
from minisgl.scheduler.prefill import ChunkedReq


class TraceLLM(ReplayLLM):
    def reset_replay(self, rows, trace):
        super().reset_replay(rows, trace)
        self.batch_index = 0
        self.observations = {}
        self.current_batch = None

    def offline_receive_msg(self, blocking=False):
        if self.batch_index == len(self.script):
            assert len(self.done) == len(self.rows)
            raise RequestAllFinished()
        return [UserMsg(uid=u, input_ids=self.inputs[u], sampling_params=SamplingParams(
            temperature=0, ignore_eos=True, max_tokens=self.row_map[u]['max_tokens']))
            for u in self.script[self.batch_index]['arrivals']]

    def _schedule_next_batch(self):
        expected = self.script[self.batch_index]
        if expected['phase'] == 'prefill':
            batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        else:
            batch = self.decode_manager.schedule_next_batch()
        assert batch is not None
        assert [r.uid for r in batch.reqs] == expected['uids']
        assert sum(r.extend_len for r in batch.reqs) == expected['input_tokens']
        forward = self._prepare_batch(batch)
        self.current_batch = batch
        self.batch_index += 1
        return forward


def script_from_events(events):
    script, arrivals = [], []
    for e in events:
        if e['event'] == 'request':
            arrivals.append(e['uid'])
        elif e['event'] == 'batch':
            script.append(dict(arrivals=arrivals, phase=e['phase'], uids=e['uids'], input_tokens=e['input_tokens']))
            arrivals = []
    assert not arrivals
    return script


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pools', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--graph', type=int, choices=(0, 8), required=True)
    a = p.parse_args()
    selection = json.loads(a.selection.read_text())
    pools = json.loads(a.pools.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    llm = TraceLLM(a.model, cuda_graph_max_bs=a.graph, cache_type='naive', num_page_override=65536,
                  page_size=1, max_running_req=8, max_seq_len_override=8192, max_extend_tokens=1024)
    original = llm.engine.sampler.sample
    targets = {int(u): i for u, i in selection['targets'].items()}
    def sample(logits, args):
        for i, req in enumerate(llm.current_batch.reqs):
            if isinstance(req, ChunkedReq) or req.uid not in targets:
                continue
            index = len(llm.outputs[req.uid])
            if index != targets[req.uid]:
                continue
            values = logits[i].float().cpu()
            top = torch.topk(values, 5)
            prefix = llm.row_map[req.uid]['input_ids'] + llm.outputs[req.uid]
            llm.observations[str(req.uid)] = dict(index=index, prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                batch_size=llm.current_batch.size, padded_size=llm.current_batch.padded_size,
                phase=llm.current_batch.phase, dtype=str(logits.dtype), argmax=int(values.argmax()),
                top_ids=top.indices.tolist(), top_values=top.values.tolist(),
                logits_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest())
            torch.save(values, a.output / f'{llm.label}_uid{req.uid}.pt')
        return original(logits, args)
    llm.engine.sampler.sample = sample
    results = []
    try:
        for key in ('baseline', 'alternative'):
            name = selection[key]
            record = json.loads((a.source / (name + '.json')).read_text()); job = record['job']
            rows = make_workload(pools[job['split']], job['scenario'], job['count'], job['rate'], job['seed'])
            assert workload_hash(rows) == record['workload_sha256']
            llm.row_map = {r['uid']: r for r in rows}
            llm.script = script_from_events([json.loads(x) for x in (a.source / (name + '.jsonl')).read_text().splitlines()])
            expected = json.loads((a.source / (name + '_outputs.json')).read_text())
            for repeat in range(2):
                llm.label = f'{key}_r{repeat}'
                llm.replay(rows, False)
                actual = {str(u): tokens for u, tokens in llm.outputs.items()}
                result = dict(case=key, repeat=repeat, graph=a.graph, batches=llm.batch_index,
                    identical_to_recorded=actual == expected, observations=llm.observations, outputs=actual)
                results.append(result)
                (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps({k:result[k] for k in ('case','repeat','graph','batches','identical_to_recorded')}), flush=True)
    finally:
        llm.shutdown()


if __name__ == '__main__':
    main()
