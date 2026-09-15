"""Capture both real policies with Nsight after excluding model load and warmup."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
for policy in ('prefill_first','wait_time'):
    out=a.output/policy;out.mkdir(exist_ok=True)
    command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
        '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
        '--output',str(out/'capture'),sys.executable,str(Path(__file__).with_name('profile_schedule.py')),
        '--model',a.model,'--pools','study/stage04_system_evaluation/prepared/pools.json',
        '--job',f'study/stage04_system_evaluation/runs/main/private/eval_mixed_steady_near_{policy}_b1024_n24_r4.json',
        '--output',str(out)]
    with (out/'profile.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=true','--output',str(out/'capture.sqlite'),str(out/'capture.nsys-rep')],check=True,stdout=subprocess.DEVNULL)
    for report in ('cuda_gpu_kern_sum','nvtx_gpu_proj_sum','nvtx_gpu_proj_trace'):
        with (out/(report+'.log')).open('w') as log:
            subprocess.run(['nsys','stats','--report',report,'--format','csv','--output',str(out/'stats'),str(out/'capture.sqlite')],check=True,stdout=log,stderr=subprocess.STDOUT)
    print('PASS Nsight '+policy,flush=True)
