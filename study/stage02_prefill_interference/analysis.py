"""Request-group metrics and CPU-observed prefill spans inside the worst short-request gap."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage01_measurement'))
from metrics import percentiles, summarize


def analyze(events):
    result=summarize(events)
    short={e['uid'] for e in events if e['event']=='request' and e['uid']<4}
    starts={e['uid']:e['arrival'] for e in events if e['event']=='request'}
    tokens={uid:[] for uid in starts}
    for e in events:
        if e['event']=='token': tokens[e['uid']].append(e['t'])
    gaps=[dict(uid=uid,start=a,end=b,seconds=b-a) for uid in short
          for a,b in zip(tokens[uid],tokens[uid][1:])]
    result['short_itl_s']=percentiles([g['seconds'] for g in gaps])
    result['short_ttft_s']=percentiles([tokens[u][0]-starts[u] for u in short])
    result['long_ttft_s']=percentiles([ts[0]-starts[u] for u,ts in tokens.items() if u not in short])
    worst=max(gaps,key=lambda g:g['seconds']) if gaps else None
    if worst:
        spans=[e for e in events if e['event']=='execution' and e['phase']=='prefill'
               and e['start']>=worst['start'] and e['end']<=worst['end']]
        decodes=[e for e in events if e['event']=='batch' and e['phase']=='decode'
                 and worst['start']<e['t']<worst['end'] and worst['uid'] in e['uids']]
        worst=dict(**worst,prefill_batches=len(spans),prefill_tokens=sum(e['input_tokens'] for e in spans),
                   prefill_cpu_span_s=sum(e['end']-e['start'] for e in spans),
                   decode_batches_in_gap=len(decodes))
    result['worst_short_gap']=worst
    return result
