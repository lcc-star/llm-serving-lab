"""Summarize real Nsight kernel evidence and anonymous GPU event spans."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def blocking_chain(batches):
    best={'count':0,'start_ms':0.,'end_ms':0.,'span_ms':0.}
    chain=[]
    for b in batches:
        if b['phase']=='prefill' and b['decode_waiting']>0:
            chain.append(b)
            span=b['end_ms']-chain[0]['start_ms']
            if span>best['span_ms']:
                best=dict(count=len(chain),start_ms=chain[0]['start_ms'],end_ms=b['end_ms'],span_ms=span)
        else:chain=[]
    return best


def summarize(root):
    records=[]
    for policy in ('prefill_first','wait_time'):
        folder=root/policy
        raw=json.loads((folder/'gpu_timeline.json').read_text())
        assert raw['completed']==24 and raw['output_tokens']==4608
        assert (folder/'capture.nsys-rep').is_file()
        with sqlite3.connect('file:'+str(folder/'capture.sqlite')+'?mode=ro',uri=True) as db:
            columns={r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
            name='demangledName' if 'demangledName' in columns else 'shortName'
            kernels=[dict(name=row[0],instances=row[1],total_ms=row[2]/1e6) for row in db.execute(
                f'SELECT s.value, COUNT(*), SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.{name}=s.id GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC')]
            graph_count,graph_ns=db.execute('SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()
        assert kernels
        event_batches=raw['batches']
        with (folder/'stats_nvtx_gpu_proj_trace.csv').open() as f:
            ranges=[r for r in csv.DictReader(f) if r['Name'] in (':stage5_prefill',':stage5_decode')]
        ranges.sort(key=lambda r:int(r['Orig Start (ns)']))
        assert len(ranges)==len(event_batches)
        origin=min(int(r['Projected Start (ns)']) for r in ranges)
        batches=[]
        for r,b in zip(ranges,event_batches):
            assert r['Name']==':stage5_'+b['phase']
            start=(int(r['Projected Start (ns)'])-origin)/1e6
            batches.append(dict(phase=b['phase'],requests=b['requests'],input_tokens=b['input_tokens'],decode_waiting=b['decode_waiting'],start_ms=start,end_ms=start+int(r['Projected Duration (ns)'])/1e6))
        phases={phase:dict(batches=sum(b['phase']==phase for b in batches),
            nvtx_gpu_projection_sum_ms=sum(b['end_ms']-b['start_ms'] for b in batches if b['phase']==phase)) for phase in ('prefill','decode')}
        records.append(dict(policy=policy,completed=raw['completed'],workload_sha256=raw['workload_sha256'],
            scheduler_sha256=raw['scheduler_sha256'],explicit_kernel_instances=sum(k['instances'] for k in kernels),
            explicit_kernel_time_sum_ms=sum(k['total_ms'] for k in kernels),graph_instances=graph_count,graph_time_sum_ms=graph_ns/1e6,top_kernels=kernels[:10],phases=phases,
            longest_blocking_prefill_chain=blocking_chain(batches),batches=batches))
    assert records[0]['workload_sha256']==records[1]['workload_sha256']
    assert records[0]['scheduler_sha256']==records[1]['scheduler_sha256']
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=summarize(a.runs);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'profile_summary.json').write_text(json.dumps(dict(records=records,note='Nsight runs are attribution evidence, not formal latency/throughput benchmarks. NVTX GPU projections span first to last associated operation. Explicit kernels exclude graph internals; graph spans are reported separately.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'prefill':'#d88724','decode':'#3286bb'}
    fig,axes=plt.subplots(2,2,figsize=(13,6),layout='constrained')
    for row,r in enumerate(records):
        chain=r['longest_blocking_prefill_chain']
        for col,ax in enumerate(axes[row]):
            lo,hi=(0,max(b['end_ms'] for b in r['batches'])) if col==0 else (max(0,chain['start_ms']-50),chain['end_ms']+50)
            for phase,y in [('prefill',1),('decode',0)]:
                intervals=[(b['start_ms']/1000,(b['end_ms']-b['start_ms'])/1000) for b in r['batches'] if b['phase']==phase and b['end_ms']>=lo and b['start_ms']<=hi]
                ax.broken_barh(intervals,(y-.3,.6),facecolors=colors[phase])
            ax.set(xlim=(lo/1000,hi/1000),yticks=[0,1],yticklabels=['decode','prefill'],xlabel='Seconds from first projected GPU operation',title=r['policy']+(' / full trace' if col==0 else ' / longest blocking prefill chain'))
            ax.grid(axis='x',alpha=.2)
    fig.suptitle('Nsight NVTX-to-GPU projections: first to last GPU operation per forward')
    for suffix in ('png','svg'):fig.savefig(a.output/('gpu_timeline.'+suffix),dpi=150)
    f=a.output/'gpu_timeline.svg';f.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('batches','top_kernels')} for r in records],indent=2))


if __name__=='__main__':main()
