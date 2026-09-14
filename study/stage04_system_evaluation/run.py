"""Staged evaluation: calibration, fixed-budget matrix, tune-only sweep, confirmation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from workloads import SCENARIOS, POLICIES, page_capacity

ROOT = Path(__file__).resolve().parent


def job(scenario, policy, level, rate, repeat, split, count=24, budget=1024, seed_base=50000):
    seed = seed_base + SCENARIOS.index(scenario)*100 + repeat
    return dict(name=f'{split}_{scenario}_{level}_{policy}_b{budget}_n{count}_r{repeat}',
                scenario=scenario, policy=policy, level=level, rate=rate, repetition=repeat,
                split=split, count=count, budget=budget, seed=seed, pages=page_capacity(scenario))


def execute(jobs, args, label):
    for pages in sorted({j['pages'] for j in jobs}, reverse=True):
        selected = [j for j in jobs if j['pages']==pages]
        path = args.output / f'{label}_jobs_{pages}.json'
        path.write_text(json.dumps(selected,indent=2)+'\n')
        log = args.output / f'{label}_{pages}.log'
        print(f'{label}: {len(selected)} jobs, KV pages={pages}', flush=True)
        with log.open('a') as stream:
            subprocess.run([sys.executable,str(ROOT/'worker.py'),'--model',args.model,
                            '--pools',str(args.pools),'--jobs',str(path),
                            '--output',str(args.output/'private'),'--pages',str(pages)],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    return [json.loads((args.output/'private'/(j['name']+'.json')).read_text()) for j in jobs]


def pilot(args):
    saturation = [job(s,'prefill_first','saturation',0,0,'tune',seed_base=10000) for s in SCENARIOS]
    measured = execute(saturation,args,'pilot_saturation')
    rates = {r['job']['scenario']:r['job']['count']/r['seconds'] for r in measured}
    calibration = {}
    for attempt in range(3):
        todo = [s for s in SCENARIOS if s not in calibration]
        if not todo:
            break
        probes = [job(s,'prefill_first',f'probe{attempt}',rates[s]*1.2,0,'tune',seed_base=11000+attempt*1000) for s in todo]
        for r in execute(probes,args,f'pilot_probe{attempt}'):
            scenario = r['job']['scenario']
            pressure = r['metrics']['max_pending'] >= 2 or r['metrics']['max_decode_running'] >= 7
            if pressure or attempt==2:
                anchor = r['job']['rate']
                calibration[scenario] = dict(anchor_rate=anchor,pressure_observed=pressure,
                                              low=anchor*.35, medium=anchor*.65, near=anchor*.95,
                                              probe_job=r['job']['name'])
            else:
                rates[scenario] *= 1.5
    (args.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    return calibration


def matrix_jobs(calibration, repeats):
    jobs=[]
    for rep in range(repeats):
        configs=[(s,l,p) for s in SCENARIOS for l in ('low','medium','near') for p in POLICIES]
        configs=configs[rep:]+configs[:rep]
        if rep%2: configs.reverse()
        jobs += [job(s,p,l,calibration[s][l],rep,'eval') for s,l,p in configs]
    return jobs


def sweep(args, calibration):
    jobs=[]
    for rep in range(args.repeats):
        for s in ('mixed_steady','long_burst'):
            configs=[('prefill_first',1024),('wait_time',256),('wait_time',1024),('wait_time',4096)]
            if rep%2:configs.reverse()
            jobs += [job(s,p,'sweep',calibration[s]['near'],rep,'tune',budget=b,seed_base=30000) for p,b in configs]
    records=execute(jobs,args,'sweep')
    choices={}
    for s in ('mixed_steady','long_burst'):
        rows=[r for r in records if r['job']['scenario']==s]
        baseline=statistics.median(r['metrics']['output_tokens_per_s'] for r in rows if r['job']['policy']=='prefill_first')
        candidates=[]
        for budget in (256,1024,4096):
            runs=[r for r in rows if r['job']['policy']=='wait_time' and r['job']['budget']==budget]
            throughput=statistics.median(r['metrics']['output_tokens_per_s'] for r in runs)
            p99=statistics.median(r['metrics']['short_itl_s']['p99'] for r in runs)
            if throughput >= .95*baseline:
                candidates.append((p99,budget))
        choices[s]=dict(budget=min(candidates)[1] if candidates else 1024,
                        throughput_constraint_met=bool(candidates),
                        rule='Minimize tune-set median short P99 ITL subject to throughput >= 95% of prefill_first/1024; fallback=1024.')
    (args.output/'budget_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    return choices


def confirm(args, calibration, choices):
    jobs=[]
    for rep in range(args.repeats):
        for s in ('mixed_steady','long_burst'):
            configs=[('prefill_first',1024),('decode_first',1024),('wait_time',1024)]
            chosen=choices[s]['budget']
            if chosen!=1024:configs.append(('wait_time',chosen))
            if rep%2: configs.reverse()
            jobs += [job(s,p,'confirm',calibration[s]['near'],rep,'eval',count=80,budget=b,seed_base=70000) for p,b in configs]
    return execute(jobs,args,'confirm')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--pools',type=Path,default=ROOT/'prepared/pools.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/main')
    parser.add_argument('--stage',choices=('pilot','matrix','sweep','confirm','all'),default='all')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<5:raise ValueError('Formal evaluation requires at least five repeats')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ('pilot','all'):
        calibration=pilot(args)
    else:
        calibration=json.loads((args.output/'calibration.json').read_text())
    if args.stage in ('matrix','all'):
        execute(matrix_jobs(calibration,args.repeats),args,'matrix')
    if args.stage in ('sweep','all'):
        choices=sweep(args,calibration)
    if args.stage in ('confirm','all'):
        if args.stage=='confirm':choices=json.loads((args.output/'budget_choices.json').read_text())
        confirm(args,calibration,choices)
    print('Completed requested evaluation stage: '+args.stage,flush=True)


if __name__=='__main__':main()
