"""Independently recompute request-group metrics and plot the observed policy tradeoff."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage02_prefill_interference'))
from analysis import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    root=args.directory;data=json.loads((root/'summary.json').read_text());groups=[]
    for scenario in ('short_only','long_burst','sustained'):
        for policy in data.get('policies', ('prefill_first','decode_first','alternating')):
            runs=[r for r in data['runs'] if r['scenario']==scenario and r['policy']==policy]
            waits=[]
            for r in runs:
                events=[json.loads(l) for l in (root/f'{scenario}_{policy}_repeat{r["repetition"]}.jsonl').read_text().splitlines()]
                assert analyze(events)==r['metrics']
                arrivals={e['uid']:e['arrival'] for e in events if e['event']=='request'}
                first={}
                for e in events:
                    if e['event']=='batch':
                        for uid in e['uids']:first.setdefault(uid,e['t'])
                waits.append(max((first[u]-t for u,t in arrivals.items() if u>=4),default=0)*1000)
            def med(fn):return statistics.median(fn(r) for r in runs)
            groups.append(dict(scenario=scenario,policy=policy,repeats=len(runs),
                short_itl_p50_ms=med(lambda r:r['metrics']['short_itl_s']['p50']*1000),
                short_itl_p95_ms=med(lambda r:r['metrics']['short_itl_s']['p95']*1000),
                short_max_itl_ms=med(lambda r:r['metrics']['short_itl_s']['max']*1000),
                short_max_itl_range_ms=[min(r['metrics']['short_itl_s']['max']*1000 for r in runs),max(r['metrics']['short_itl_s']['max']*1000 for r in runs)],
                long_ttft_p50_ms=med(lambda r:r['metrics']['long_ttft_s']['p50']*1000) if scenario!='short_only' else None,
                long_ttft_max_ms=med(lambda r:r['metrics']['long_ttft_s']['max']*1000) if scenario!='short_only' else None,
                long_max_first_schedule_wait_ms=statistics.median(waits),
                throughput=med(lambda r:r['metrics']['output_tokens_per_s']),
                output_hashes=len({r['sha256'] for r in runs})))
    (root/'comparison.json').write_text(json.dumps(groups,indent=2)+'\n')
    colors={'prefill_first':'#d88724','decode_first':'#3286bb','alternating':'#27935a','wait_time':'#7952b3'}
    height=120+140*len(data.get('policies', ('prefill_first','decode_first','alternating')))
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="14">',
         '<text x="20" y="25">Stage 3: medians across runs; CPU-observed milliseconds (separate scales per panel)</text>']
    for panel,(field,title) in enumerate([('short_max_itl_ms','Maximum short-request ITL'),('long_ttft_p50_ms','Long-request median TTFT')]):
        chosen=[g for g in groups if g['scenario']!='short_only'];maximum=max(g[field] for g in chosen)
        left=20+panel*590
        svg.append(f'<text x="{left}" y="60">{title}</text>')
        for i,g in enumerate(chosen):
            y=90+i*70;width=300*g[field]/maximum
            svg.append(f'<text x="{left}" y="{y}">{g["scenario"]} / {g["policy"]}</text><rect x="{left}" y="{y+8}" width="{width}" height="20" fill="{colors[g["policy"]]}"/><text x="{left+width+10}" y="{y+24}">{g[field]:.2f}</text>')
    svg.append('</g></svg>');(root/'comparison.svg').write_text('\n'.join(svg))
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
