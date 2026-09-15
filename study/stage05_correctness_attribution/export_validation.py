"""Verify complete lifecycle and post-fix evidence; publish aggregate records only."""
import argparse
import hashlib
import json
import re
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[];extra=[]
for cache,overlap in [('naive',0),('naive',1),('radix',0),('radix',1)]:
    name=f'{cache}_overlap{overlap}.json'
    r=json.loads((a.runs/'lifecycle'/name).read_text());e=json.loads((a.runs/'extra'/name).read_text())
    assert r['status']==e['status']=='ok'
    assert len(r['results'])==48 and len(e['results'])==9
    assert not list((a.runs/'lifecycle').glob('*.failed.json'))
    for policy in ('prefill_first','decode_first','wait_time'):
        rows=[x for x in r['results'] if x['policy']==policy]
        assert len(rows)==16
        assert next(x for x in rows if x['case']=='recycle')['waves']>=100
        shared=next(x for x in e['results'] if x['policy']==policy and x['case']=='shared')
        if cache=='radix':assert shared['shared_verified']
        assert next(x for x in e['results'] if x['policy']==policy and x['case']=='cancel_inflight_decode')['decode_launched']
    if cache=='radix' and overlap:assert r['soak_seconds']>=600 and r['soak_completed']>0
    # Historical field counts cached chunk continuations too, not only cache hits.
    for row in r['results']:
        if 'cache_hits' in row:row['prefill_batches_with_cached_tokens']=row.pop('cache_hits')
    records.append(r);extra.append(e)
raw=json.loads((a.runs/'graph_after_fix/results.json').read_text())
assert len(raw)==4 and all(x['identical_to_recorded'] for x in raw)
for case in ('baseline','alternative'):
    left,right=[x for x in raw if x['case']==case]
    assert left['outputs']==right['outputs'] and left['observations']==right['observations']
regressions=[{k:x[k] for k in ('case','repeat','graph','batches','identical_to_recorded')} for x in raw]
scheduler_hash=hashlib.sha256(Path('python/minisgl/scheduler/scheduler.py').read_bytes()).hexdigest()
assert all(r['provenance']['source_sha256']['python/minisgl/scheduler/scheduler.py']==scheduler_hash for r in records)
cache_key='python/minisgl/scheduler/cache.py'
cache_hash=hashlib.sha256(Path(cache_key).read_bytes()).hexdigest()
assert all(r['provenance']['source_sha256'].get(cache_key)==cache_hash for r in records if r['cache']=='radix'), 'Radix evidence must match final cache source'
assert all(r['provenance']['source_sha256'][cache_key]==cache_hash and r['provenance']['source_sha256']['python/minisgl/scheduler/scheduler.py']==scheduler_hash for r in extra)
cpu=json.loads((a.runs/'cpu_validation.json').read_text())
assert all(r['exit_code']==0 for r in cpu)
cpu_counts=[dict(suite=r['suite'],passed=int(re.search(r'Ran (\d+) tests',r['summary']).group(1))) for r in cpu]
pytest_log=(a.runs/'cache_allocation_pytest.txt').read_text()
assert re.search(r'6 passed in',pytest_log) and 'FAILED' not in pytest_log
cpu_counts.append(dict(suite='tests/core/test_cache_allocate.py',passed=6))
result=dict(status='passed',cpu_tests=cpu_counts,cpu_passed=sum(r['passed'] for r in cpu_counts),lifecycle=records,extra=extra,post_fix_fixed_trace=regressions,
    named_lifecycle_cases=180,extra_cases=36,recycle_waves=sum(x['waves'] for r in records for x in r['results'] if x['case']=='recycle'),
    recycle_completed=sum(x['completed'] for r in records for x in r['results'] if x['case']=='recycle'),
    soak_completed=sum(r['soak_completed'] for r in records),
    note='Forced sampler validates lifecycle, not model semantics. Finite ten-minute soak is not a proof of unbounded leak freedom. Evicted-page counters include explicit between-case cache clearing.')
def reject_private_payloads(value):
    if isinstance(value,dict):
        forbidden={'uid','input_ids','outputs','token_ids','top_ids','prefix_sha256','mapping_sha256','argmax','tensors','prompt','messages'}
        assert not forbidden.intersection(value), 'Private payload in public summary'
        for child in value.values():reject_private_payloads(child)
    elif isinstance(value,list):
        for child in value:reject_private_payloads(child)
reject_private_payloads(result)
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'lifecycle_summary.json').write_text(json.dumps(result,indent=2)+'\n')
before=json.loads((a.runs/'lifecycle_before.failed.json').read_text())
(a.output/'regression_before.json').write_text(json.dumps({k:before[k] for k in ('status','cache','overlap','case','error_type','error')},indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('lifecycle','extra','post_fix_fixed_trace')},indent=2))
