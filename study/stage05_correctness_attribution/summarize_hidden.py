"""Compare logical positions and hidden tensors; export numeric aggregates only."""
import argparse
import json
from pathlib import Path
import torch


def delta(a, b):
    assert a.shape == b.shape
    diff = (a.float() - b.float()).abs()
    return dict(equal=torch.equal(a, b), shape=list(a.shape),
                max_abs=diff.max().item(), mean_abs=diff.mean().item(),
                different_elements=int(torch.count_nonzero(diff)))


def reference_attention(q, k, v):
    """One decode query, already rotated Q/K, contiguous GQA head groups."""
    assert q.shape[0] == 1 and k.shape == v.shape
    heads, width = q.shape[-2:]
    kv_heads = k.shape[-2]
    assert heads % kv_heads == 0
    query = q.double().reshape(kv_heads, heads // kv_heads, width)
    scores = torch.einsum('hgd,nhd->hgn', query, k.double()) / width ** 0.5
    return torch.einsum('hgn,nhd->hgd', scores.softmax(-1), v.double()).reshape(1, -1)


def prefill_tensor(snapshots, name):
    chunks = sorted((s for s in snapshots if s['phase'] == 'prefill'), key=lambda s: s['start'])
    end = 0
    for s in chunks:
        assert s['start'] == end, 'Missing or overlapping logical positions'
        assert s['tensors'][name].shape[0] == s['end'] - s['start']
        end = s['end']
    return torch.cat([s['tensors'][name] for s in chunks])


def identical_snapshots(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        # Physical page IDs may be reused differently; compare logical tensor contents.
        for k in set(x) - {'tensors', 'mapping_sha256'}:
            if x[k] != y[k]:
                return False
        if x['tensors'].keys() != y['tensors'].keys():
            return False
        if not all(torch.equal(x['tensors'][k], y['tensors'][k]) for k in x['tensors']):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = json.loads((a.runs / 'results.json').read_text())
    assert len(records) == 4 and all(r['probe_preserves_eager_output'] for r in records)
    data = {key: [torch.load(a.runs / f'{key}_r{i}.pt', weights_only=True) for i in range(2)]
            for key in ('baseline', 'alternative')}
    repeatability = {key: identical_snapshots(*runs) for key, runs in data.items()}
    assert all(repeatability.values())
    left, right = data['baseline'][0], data['alternative'][0]
    prefill = {}
    for name in left[0]['tensors']:
        prefill[name] = delta(prefill_tensor(left, name), prefill_tensor(right, name))
    x, y = next(s for s in left if s['phase'] == 'decode'), next(s for s in right if s['phase'] == 'decode')
    assert x['prefix_sha256'] == y['prefix_sha256']
    decode = {name: delta(value, y['tensors'][name]) for name, value in x['tensors'].items()}
    head = {}
    for name, s in (('baseline', x), ('alternative', y)):
        fp64 = s['candidate_fp64']
        head[name] = dict(candidate_scores=s['candidate_scores'], candidate_fp64=fp64,
            fp64_rounded_bf16=torch.tensor(fp64, dtype=torch.float64).bfloat16().float().tolist())
    attention_reference = {}
    if 'layer0/rotated_q' in x['tensors']:
        references = []
        for label, snapshot in (('baseline', x), ('alternative', y)):
            tensors = snapshot['tensors']
            ref = reference_attention(tensors['layer0/rotated_q'],
                torch.cat([tensors['layer0/history_k'], tensors['layer0/stored_k']]),
                torch.cat([tensors['layer0/history_v'], tensors['layer0/stored_v']]))
            actual = tensors['layer0/attention'].double()
            error = (actual - ref).abs()
            attention_reference[label] = dict(max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
                different_from_rounded_reference=int(torch.count_nonzero(actual != ref.bfloat16().double())))
            references.append(ref)
        attention_reference['identical_reference_outputs'] = torch.equal(*references)
    result = dict(runs=4, repeatability=repeatability, probe_preserves_eager_output=True,
        mapping_checks=sum(r['mapping_checks'] for r in records), kv_store_checks=sum(r['kv_store_checks'] for r in records),
        same_decode_prefix=True, output_index=x['output_index'],
        baseline_prefill_chunks=[[s['start'],s['end']] for s in left if s['phase']=='prefill'],
        alternative_prefill_chunks=[[s['start'],s['end']] for s in right if s['phase']=='prefill'],
        prefill=prefill, decode=decode, head=head, attention_reference=attention_reference,
        note='Naive-cache ownership/write checks at the selected request steps only. Equal logical tokens do not prove global cache correctness; no performance claim.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('prefill','decode')}, indent=2))
    print('Prefill boundary differences:',json.dumps({k:v['max_abs'] for k,v in prefill.items()}))
    print('Decode layer0 boundaries:',json.dumps({k:v['max_abs'] for k,v in decode.items() if k.startswith('layer0/') or k.startswith('final_norm')}))


if __name__ == '__main__':
    main()
