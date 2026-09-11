"""Stage 2: only scheduling prefill token budget varies; original policy is preserved."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage01_measurement'))
from replay import ReplayLLM, torch
from analysis import analyze


class InterferenceLLM(ReplayLLM):
    def _forward(self, forward):
        b=forward.batch
        self.span=dict(event='execution',start=self.now(),phase=b.phase,
                       uids=[r.uid for r in b.reqs],input_tokens=sum(r.extend_len for r in b.reqs))
        return super()._forward(forward)

    def _process_last_data(self, data):
        super()._process_last_data(data)
        if data is not None:
            self.events.append(dict(**self.span,end=self.now()))


def workloads():
    rng=random.Random(2026)
    rows=[dict(uid=i,arrival=0.0 if i<4 else .3,
               input_ids=[rng.randrange(100,10000) for _ in range(128 if i<4 else 4096)],
               max_tokens=256 if i<4 else 128) for i in range(8)]
    return {'short_only':rows[:4], 'long_burst':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    assert args.repeats>0
    args.output.mkdir(parents=True,exist_ok=True)
    cases=workloads()
    (args.output/'workloads.json').write_text(json.dumps(cases)+'\n')
    llm=InterferenceLLM(args.model,cuda_graph_max_bs=8,cache_type='naive',num_page_override=32768,
        page_size=1,max_running_req=8,max_seq_len_override=8192,max_extend_tokens=4096)
    runs=[]
    try:
        for budget in (256,1024,4096):
            llm.prefill_budget=budget
            llm.replay(cases['long_burst'],True)  # each budget warmed, excluded
        configs=[(s,b) for s in cases for b in (256,1024,4096)]
        for rep in range(args.repeats):
            order=configs[rep%len(configs):]+configs[:rep%len(configs)]
            if rep%2: order=list(reversed(order))
            for scenario,budget in order:
                llm.prefill_budget=budget
                record=llm.replay(cases[scenario],True)
                events=llm.events
                expected=sum(r['max_tokens'] for r in cases[scenario])
                assert record['output_tokens']==expected
                metrics=analyze(events)
                assert metrics['incomplete']==0
                assert all(e['input_tokens']<=budget for e in events if e['event']=='batch' and e['phase']=='prefill')
                if scenario=='long_burst':
                    # All short requests must have started decoding before the fixed burst.
                    assert all(any(e['event']=='token' and e['uid']==uid and e['t']<.3 for e in events) for uid in range(4))
                name=f'{scenario}_budget{budget}_repeat{rep}'
                (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
                (args.output/f'{name}_outputs.json').write_text(json.dumps(llm.outputs)+'\n')
                row=dict(scenario=scenario,budget=budget,repetition=rep,**record,metrics=metrics)
                runs.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
        result=dict(runs=runs,model=Path(args.model).name,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                    graph=True,overlap=False,cache='naive',kv_pages=32768,page_size=1,
                    max_running_requests=8,max_seq_len=8192,burst_arrival=.3,
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                    workload_sha256=hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
                    note='CPU-observed spans; no GPU event timings. Different schedules may change BF16 outputs; no output equality claimed.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        llm.shutdown()


if __name__=='__main__': main()
