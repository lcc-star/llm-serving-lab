"""Recompute evidence summaries and a standalone comparison SVG."""
import argparse
import json
from pathlib import Path
import statistics
from analysis import analyze


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    root=args.directory
    data=json.loads((root/'summary.json').read_text())
    groups=[]
    for scenario in ('short_only','long_burst'):
        for budget in (256,1024,4096):
            runs=[r for r in data['runs'] if r['scenario']==scenario and r['budget']==budget]
            assert runs
            for r in runs:
                f=root/f'{scenario}_budget{budget}_repeat{r["repetition"]}.jsonl'
                events=[json.loads(l) for l in f.read_text().splitlines()]
                assert analyze(events)==r['metrics']
            def median(fn): return statistics.median(fn(r) for r in runs)
            groups.append(dict(scenario=scenario,budget=budget,runs=len(runs),
                short_itl_p50_ms=median(lambda r:r['metrics']['short_itl_s']['p50']*1000),
                short_itl_p95_ms=median(lambda r:r['metrics']['short_itl_s']['p95']*1000),
                worst_short_itl_ms=median(lambda r:r['metrics']['short_itl_s']['max']*1000),
                worst_short_itl_range_ms=[min(r['metrics']['short_itl_s']['max']*1000 for r in runs),max(r['metrics']['short_itl_s']['max']*1000 for r in runs)],
                long_ttft_p50_ms=median(lambda r:r['metrics']['long_ttft_s']['p50']*1000) if scenario=='long_burst' else None,
                throughput=median(lambda r:r['metrics']['output_tokens_per_s']),
                worst_gap_prefill_batches=median(lambda r:r['metrics']['worst_short_gap']['prefill_batches']),
                worst_gap_prefill_tokens=median(lambda r:r['metrics']['worst_short_gap']['prefill_tokens']),
                worst_gap_prefill_cpu_ms=median(lambda r:r['metrics']['worst_short_gap']['prefill_cpu_span_s']*1000),
                unique_output_hashes=len({r['sha256'] for r in runs})))
    (root/'comparison.json').write_text(json.dumps(groups,indent=2)+'\n')
    maximum=max(g['worst_short_itl_ms'] for g in groups)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="370" viewBox="0 0 1100 370">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="15">',
         '<text x="20" y="30">Median of per-run maximum short-request ITL (CPU observed, milliseconds)</text>']
    for i,g in enumerate(groups):
        y=65+i*45; width=650*g['worst_short_itl_ms']/maximum
        label=f'{g["scenario"]} / {g["budget"]}'
        color='#d88724' if g['scenario']=='long_burst' else '#3286bb'
        svg.append(f'<text x="20" y="{y+17}">{label}</text><rect x="240" y="{y}" width="{width}" height="25" fill="{color}"/><text x="{250+width}" y="{y+17}">{g["worst_short_itl_ms"]:.2f} ms</text>')
    svg.append('</g></svg>')
    (root/'comparison.svg').write_text('\n'.join(svg))
    # Zoom into the worst short-request gap of repeat 0, separately for each budget.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="350" viewBox="0 0 1200 350">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="14">',
         '<text x="20" y="25">Worst gap, repeat 0: orange = prefill CPU span; green = bounding short-request tokens</text>',
         '<text x="20" y="48">Each row normalized to its own gap; labels show actual elapsed time. Not GPU kernel timing.</text>']
    for i,budget in enumerate((256,1024,4096)):
        r=next(r for r in data['runs'] if r['scenario']=='long_burst' and r['budget']==budget and r['repetition']==0)
        gap=r['metrics']['worst_short_gap']; y=90+i*80
        events=[json.loads(l) for l in (root/f'long_burst_budget{budget}_repeat0.jsonl').read_text().splitlines()]
        def x(t): return 230+850*(t-gap['start'])/gap['seconds']
        svg.append(f'<text x="20" y="{y+17}">budget {budget}, uid {gap["uid"]}</text>')
        for e in events:
            if e['event']=='execution' and e['phase']=='prefill' and gap['start']<=e['start'] and e['end']<=gap['end']:
                svg.append(f'<rect x="{x(e["start"])}" y="{y}" width="{max(.5,x(e["end"])-x(e["start"]))}" height="22" fill="#d88724" stroke="white"/>')
        svg.append(f'<circle cx="230" cy="{y+11}" r="4" fill="green"/><circle cx="1080" cy="{y+11}" r="4" fill="green"/><text x="230" y="{y+43}">{gap["seconds"]*1000:.2f} ms; {gap["prefill_batches"]} prefill batches; {gap["prefill_tokens"]} input tokens</text>')
    svg.append('</g></svg>')
    (root/'gap_timeline.svg').write_text('\n'.join(svg))
    print(json.dumps(groups,indent=2))


if __name__=='__main__': main()
