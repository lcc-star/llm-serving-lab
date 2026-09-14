"""Compare real Scheduler policies with the same workload and prefill budget."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage02_prefill_interference'))
from run import InterferenceLLM, workloads, torch
from analysis import analyze
from minisgl.scheduler.policy import BatchPolicy, SUPPORTED_POLICIES


class PolicyLLM(InterferenceLLM):
    def reset_replay(self,rows,trace):
        super().reset_replay(rows,trace)
        self.batch_policy=BatchPolicy(self.policy_name, decode_wait_ms=self.decode_wait_ms,
                                     prefill_wait_ms=self.prefill_wait_ms)

    def _record_scheduling_decision(self, decision):
        if self.trace:
            self.events.append(dict(event="decision", t=self.now(), **decision))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--policies', nargs='+', choices=SUPPORTED_POLICIES,
                        default=['prefill_first', 'decode_first', 'wait_time'])
    parser.add_argument('--decode-wait-ms', type=float, default=50.0)
    parser.add_argument('--prefill-wait-ms', type=float, default=200.0)
    args=parser.parse_args();assert args.repeats>0
    assert len(set(args.policies)) == len(args.policies)
    BatchPolicy('wait_time', decode_wait_ms=args.decode_wait_ms, prefill_wait_ms=args.prefill_wait_ms)
    args.output.mkdir(parents=True,exist_ok=True)
    cases=workloads()
    rng=random.Random(2027)
    sustained=[dict(r) for r in cases['short_only']]
    for r in sustained: r['max_tokens']=512
    sustained += [dict(uid=i,arrival=.3+(i-4)*.15,
                        input_ids=[rng.randrange(100,10000) for _ in range(4096)],max_tokens=128) for i in range(4,12)]
    cases['sustained']=sustained
    (args.output/'workloads.json').write_text(json.dumps(cases)+'\n')
    llm=PolicyLLM(args.model,cuda_graph_max_bs=8,cache_type='naive',num_page_override=32768,
                  page_size=1,max_running_req=8,max_seq_len_override=8192,max_extend_tokens=4096)
    llm.prefill_budget=1024
    llm.decode_wait_ms=args.decode_wait_ms
    llm.prefill_wait_ms=args.prefill_wait_ms
    runs=[]
    try:
        for policy in args.policies:
            llm.policy_name=policy;llm.replay(cases['long_burst'],True)
        configs=[(s,p) for s in cases for p in args.policies]
        for rep in range(args.repeats):
            order=configs[rep%len(configs):]+configs[:rep%len(configs)]
            if rep%2:order=list(reversed(order))
            for scenario,policy in order:
                llm.policy_name=policy
                record=llm.replay(cases[scenario],True)
                assert all(not waits for waits in llm.batch_policy.wait_since.values())
                assert record['output_tokens']==sum(r['max_tokens'] for r in cases[scenario])
                metrics=analyze(llm.events)
                assert metrics['completed']==len(cases[scenario])
                assert all(e['input_tokens']<=1024 for e in llm.events if e['event']=='batch' and e['phase']=='prefill')
                if scenario!='short_only':
                    assert all(any(e['event']=='token' and e['uid']==uid and e['t']<.3 for e in llm.events) for uid in range(4))
                name=f'{scenario}_{policy}_repeat{rep}'
                (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in llm.events))
                (args.output/f'{name}_outputs.json').write_text(json.dumps(llm.outputs)+'\n')
                row=dict(scenario=scenario,policy=policy,repetition=rep,metrics=metrics,**record)
                runs.append(row);print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
        result=dict(runs=runs,policies=args.policies,decode_wait_ms=args.decode_wait_ms,
                    prefill_wait_ms=args.prefill_wait_ms,budget=1024,graph=True,overlap=False,cache='naive',kv_pages=32768,
                    model=Path(args.model).name,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                    workload_sha256=hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
                    note='CPU observation times. Finite sustained arrivals, not proof of starvation freedom under infinite traffic.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:llm.shutdown()


if __name__=='__main__':main()
