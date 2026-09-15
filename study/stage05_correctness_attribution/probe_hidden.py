"""Natural eager replay with passive hidden/KV probes; private tensors, no timing claims."""
import argparse
import hashlib
import json
from pathlib import Path
from replay_trace import TraceLLM, script_from_events, torch, make_workload, workload_hash


def tensor_digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class ProbeLLM(TraceLLM):
    def reset_replay(self, rows, trace):
        super().reset_replay(rows, trace)
        self.snapshots = []
        self.active_probe = None
        self.mapping_checks = 0
        self.store_checks = 0

    def _forward(self, forward):
        batch = forward.batch
        offset = 0
        self.active_probe = None
        for req in batch.reqs:
            if req.uid == self.target and len(self.outputs[req.uid]) <= self.target_index:
                self.target_slice = slice(offset, offset + req.extend_len)
                self.target_req = req
                self.active_probe = dict(phase=batch.phase, start=req.cached_len, end=req.device_len,
                    output_index=len(self.outputs[req.uid]), tensors={})
                self.snapshots.append(self.active_probe)
                self.check_mapping(batch)
                break
            offset += req.extend_len
        result = super()._forward(forward)
        self.active_probe = None
        return result

    def check_mapping(self, batch):
        live = {r.uid: (r, r.cached_len) for r in self.decode_manager.running_reqs}
        live.update({p.uid: (p.chunked_req, p.chunked_req.cached_len)
                     for p in self.prefill_manager.pending_list if p.chunked_req is not None})
        live.update({r.uid: (r, r.device_len) for r in batch.reqs})
        pages = []
        for req, length in live.values():
            pages.extend(self.engine.page_table[req.table_idx, :length].cpu().tolist())
        assert len(pages) == len(set(pages)), 'Aliased live pages in naive cache'
        assert all(0 <= x < self.cache_manager.num_pages for x in pages)
        assert not set(pages).intersection(self.cache_manager.free_slots.cpu().tolist())
        req = self.target_req
        positions = batch.positions[self.target_slice].cpu()
        assert torch.equal(positions, torch.arange(req.cached_len, req.device_len, dtype=positions.dtype))
        mapping = self.engine.page_table[req.table_idx, req.cached_len:req.device_len]
        assert torch.equal(mapping, batch.out_loc[self.target_slice])
        self.active_probe['mapping_sha256'] = tensor_digest(self.engine.page_table[req.table_idx, :req.device_len].cpu())
        self.mapping_checks += 1


def install_probes(llm, candidates):
    def save(name, value, full=False):
        if llm.active_probe is None:
            return
        if not full and llm.active_probe['phase'] != 'decode':
            return
        if isinstance(value, torch.Tensor):
            llm.active_probe['tensors'][name] = value[llm.target_slice].detach().cpu().clone()
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                if isinstance(v, torch.Tensor):
                    save(name + '/' + str(i), v, full)

    def wrap(op, name, full=False):
        original = op.forward
        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            save(name, result, full)
            return result
        op.forward = forward

    model = llm.engine.model.model
    wrap(model.embed_tokens, 'embedding', True)
    for i, layer in enumerate(model.layers.op_list):
        full = i == 0
        wrap(layer.input_layernorm, f'layer{i}/input_norm', full)
        wrap(layer.self_attn.qkv_proj, f'layer{i}/qkv', full)
        wrap(layer.self_attn.attn, f'layer{i}/attention', full)
        wrap(layer.self_attn.o_proj, f'layer{i}/o_proj', full)
        wrap(layer.post_attention_layernorm, f'layer{i}/post_norm', full)
        wrap(layer.mlp.gate_up_proj, f'layer{i}/gate_up', full)
        wrap(layer.mlp.down_proj, f'layer{i}/down', full)
    wrap(model.norm, 'final_norm')
    head = llm.engine.model.lm_head
    candidate_weights = head.weight[candidates].double().cpu()
    original_head = head.forward
    def head_forward(x):
        active = llm.active_probe
        if active is not None and active['phase'] == 'decode':
            hidden = x[llm.target_slice].double().cpu()[0]
            active['candidate_fp64'] = torch.mv(candidate_weights, hidden).tolist()
        return original_head(x)
    head.forward = head_forward
    backend = llm.engine.attn_backend
    original_attention = backend.forward
    def attention(q, k, v, layer_id, batch):
        if layer_id == 0:
            save('layer0/rotated_q', q, True)
            save('layer0/rotated_k', k, True)
            save('layer0/backend_v', v, True)
        return original_attention(q, k, v, layer_id, batch)
    backend.forward = attention
    cache = llm.engine.kv_cache
    original_store = cache.store_kv
    def store(k, v, out_loc, layer_id):
        original_store(k, v, out_loc, layer_id)
        if llm.active_probe is None:
            return
        locations = out_loc[llm.target_slice].long()
        for label, values, pool in (('k', k, cache.k_cache(layer_id)), ('v', v, cache.v_cache(layer_id))):
            actual = pool.view(-1, *pool.shape[-2:])[locations].cpu()
            expected = values[llm.target_slice].reshape_as(actual).cpu()
            assert torch.equal(actual, expected), 'Stored KV differs from projected KV'
            llm.store_checks += 1
            if layer_id == 0 or llm.active_probe['phase'] == 'decode':
                llm.active_probe['tensors'][f'layer{layer_id}/stored_{label}'] = actual
        if llm.active_probe['phase'] == 'decode':
            req = llm.target_req
            indices = llm.engine.page_table[req.table_idx, :req.cached_len].long()
            for label, pool in (('k', cache.k_cache(layer_id)), ('v', cache.v_cache(layer_id))):
                llm.active_probe['tensors'][f'layer{layer_id}/history_{label}'] = pool.view(-1, *pool.shape[-2:])[indices].cpu()
    cache.store_kv = store
    original_sample = llm.engine.sampler.sample
    def sample(logits, args):
        active = llm.active_probe
        if active is not None and active['phase'] == 'decode':
            index = next(i for i, r in enumerate(llm.current_batch.reqs) if r.uid == llm.target)
            scores = logits[index].float().cpu()
            active['candidate_scores'] = scores[candidates].tolist()
            active['argmax'] = int(scores.argmax())
            prefix = llm.row_map[llm.target]['input_ids'] + llm.outputs[llm.target]
            active['prefix_sha256'] = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
        return original_sample(logits, args)
    llm.engine.sampler.sample = sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pools', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    selection = json.loads((a.previous / 'selected.json').read_text())
    target = min(selection['targets'], key=lambda u: (selection['targets'][u], int(u)))
    target_index = selection['targets'][target]
    prior = json.loads((a.previous / 'eager/results.json').read_text())
    observations = [next(r for r in prior if r['case'] == key and r['repeat'] == 0)['observations'][target]
                    for key in ('baseline', 'alternative')]
    assert observations[0]['prefix_sha256'] == observations[1]['prefix_sha256']
    candidates = [r['argmax'] for r in observations]
    assert candidates[0] != candidates[1]
    pools = json.loads(a.pools.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    llm = ProbeLLM(a.model, cuda_graph_max_bs=0, cache_type='naive', num_page_override=65536,
                   page_size=1, max_running_req=8, max_seq_len_override=8192, max_extend_tokens=1024)
    llm.target, llm.target_index = int(target), target_index
    install_probes(llm, candidates)
    results = []
    try:
        for key in ('baseline', 'alternative'):
            name = selection[key]
            job_record = json.loads((a.source / (name + '.json')).read_text()); job = job_record['job']
            rows = make_workload(pools[job['split']], job['scenario'], job['count'], job['rate'], job['seed'])
            assert workload_hash(rows) == job_record['workload_sha256']
            llm.row_map = {r['uid']: r for r in rows}
            llm.script = script_from_events([json.loads(x) for x in (a.source / (name + '.jsonl')).read_text().splitlines()])
            expected = next(r for r in prior if r['case'] == key and r['repeat'] == 0)['outputs']
            for repeat in range(2):
                llm.replay(rows, False)
                same = {str(u): t for u, t in llm.outputs.items()} == expected
                assert same, 'Probe changed natural eager output'
                torch.save(llm.snapshots, a.output / f'{key}_r{repeat}.pt')
                results.append(dict(case=key, repeat=repeat, probe_preserves_eager_output=same,
                    mapping_checks=llm.mapping_checks, kv_store_checks=llm.store_checks,
                    snapshots=len(llm.snapshots), target_output_index=target_index))
                (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(results[-1]), flush=True)
    finally:
        llm.shutdown()


if __name__ == '__main__':
    main()
