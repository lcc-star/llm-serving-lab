"""Run each cache/overlap configuration in a fresh process; preserve failures and logs locally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--waves',type=int,default=100)
p.add_argument('--soak-seconds',type=float,default=600)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]
provenance={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                    [root/'python/minisgl/scheduler/scheduler.py',root/'python/minisgl/scheduler/cache.py',Path(__file__).with_name('lifecycle.py')]}}
for cache,overlap in [('naive',0),('naive',1),('radix',0),('radix',1)]:
    name=f'{cache}_overlap{overlap}'
    command=[sys.executable,str(Path(__file__).with_name('lifecycle.py')),'--model',a.model,
        '--cache',cache,'--overlap',str(overlap),'--waves',str(a.waves),'--soak-seconds',
        str(a.soak_seconds if cache=='radix' and overlap else 0),'--output',str(a.output/(name+'.json'))]
    with (a.output/(name+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    report=json.loads((a.output/(name+'.json')).read_text());report['provenance']=provenance
    (a.output/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS '+name,flush=True)
