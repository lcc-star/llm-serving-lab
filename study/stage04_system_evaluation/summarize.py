"""Recompute private evidence and export only aggregate, allowlisted statistics."""
import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from analysis import analyze
from workloads import make_workload, workload_hash


def distribution(values):
    return dict(median=statistics.median(values),min=min(values),max=max(values),values=values)


def collect(root, pools):
    records=[]
    for path in sorted((root/'private').glob('*.json')):
        if path.name.endswith('_outputs.json'):
            continue
        record=json.loads(path.read_text())
        if record['status']!='ok':
            raise RuntimeError('Failed run exists: '+path.name)
        job=record['job']
        rows=make_workload(pools[job['split']],job['scenario'],job['count'],job['rate'],job['seed'])
        assert workload_hash(rows)==record['workload_sha256']
        events=[json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        assert analyze(events,rows)==record['metrics']
        outputs=json.loads(path.with_name(path.stem+'_outputs.json').read_text())
        assert {int(uid):len(tokens) for uid,tokens in outputs.items()}=={r['uid']:r['max_tokens'] for r in rows}
        normalized={int(uid):tokens for uid,tokens in outputs.items()}
        assert hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()==record['sha256']
        # No token IDs, source conversation IDs, prompt text or raw trace is exported.
        records.append({k:record[k] for k in ('job','status','source_commit','workload_sha256',
                                             'gpu','torch','model','seconds','output_tokens','metrics')})
    grouped=defaultdict(list)
    comparisons=defaultdict(set)
    for r in records:
        j=r['job']
        grouped[(j['split'],j['scenario'],j['level'],j['policy'],j['budget'],j['count'])].append(r)
        comparisons[(j['split'],j['scenario'],j['level'],j['repetition'],j['count'])].add(r['workload_sha256'])
    assert all(len(hashes)==1 for hashes in comparisons.values()), 'Policies did not receive identical workloads'
    groups=[]
    for key,runs in grouped.items():
        split,scenario,level,policy,budget,count=key
        metrics={}
        for field in ('ttft_s','itl_s','e2e_s','short_itl_s','long_itl_s','short_ttft_s','long_ttft_s'):
            for q in ('p50','p95','p99','max'):
                values=[r['metrics'][field][q]*1000 for r in runs if r['metrics'][field][q] is not None]
                if values:metrics[f'{field[:-2]}_{q}_ms']=distribution(values)
        for field in ('output_tokens_per_s','max_pending','max_decode_running','peak_kv_allocated_pages',
                      'min_kv_free_pages','max_first_schedule_wait_s','errors','outstanding_at_last_arrival','drain_seconds'):
            metrics[field]=distribution([r['metrics'][field] for r in runs])
        groups.append(dict(split=split,scenario=scenario,level=level,policy=policy,budget=budget,count=count,
                           repeats=len(runs),metrics=metrics,
                           short_itl_samples_per_run=[r['metrics']['short_itl_s']['count'] for r in runs]))
    return records,groups



def compare_outputs(root, records):
    baselines={}
    for r in records:
        j=r['job']
        key=(j['split'],j['scenario'],j['level'],j['repetition'],j['count'])
        if j['policy']=='prefill_first' and j['budget']==1024:
            baselines[key]=j['name']
    comparisons=[]
    for r in records:
        j=r['job']
        key=(j['split'],j['scenario'],j['level'],j['repetition'],j['count'])
        if j['policy']=='prefill_first' or key not in baselines:
            continue
        a=json.loads((root/'private'/(baselines[key]+'_outputs.json')).read_text())
        b=json.loads((root/'private'/(j['name']+'_outputs.json')).read_text())
        assert a.keys()==b.keys()
        first=[next((i for i,(x,y) in enumerate(zip(a[u],b[u])) if x!=y),None) for u in a]
        changed=[i for i in first if i is not None]
        comparisons.append(dict(scenario=j['scenario'],level=j['level'],split=j['split'],
                                policy=j['policy'],budget=j['budget'],repetition=j['repetition'],
                                requests=len(a),different_requests=len(changed),
                                first_difference_index=({k:v for k,v in distribution(changed).items() if k!='values'} if changed else None)))
    return comparisons


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--pools',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    pools=json.loads(args.pools.read_text())
    records,groups=collect(args.runs,pools)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'aggregate_runs.json').write_text(json.dumps(records,indent=2)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(groups,indent=2)+'\n')
    (args.output/'output_difference_counts.json').write_text(json.dumps(compare_outputs(args.runs,records),indent=2)+'\n')
    for name in ('calibration.json','budget_choices.json'):
        if (args.runs/name).exists():
            (args.output/name).write_text((args.runs/name).read_text())
    print(f'PASS: independently verified {len(records)} runs; exported {len(groups)} aggregate groups')


if __name__=='__main__':main()
