"""Real scheduler replay under Nsight; profiling is separate from stage-four timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage04_system_evaluation'))
from worker import EvaluationLLM,torch
from workloads import make_workload,workload_hash


class ProfileLLM(EvaluationLLM):
    def reset_replay(self,rows,trace):
        super().reset_replay(rows,trace)
        self.gpu_batches=[]
        self.epoch=torch.cuda.Event(enable_timing=True)
        self.epoch.record(self.engine.stream)

    def _forward(self,forward):
        if not self.profiling:return super()._forward(forward)
        batch=forward.batch
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        count=sum(r.extend_len for r in batch.reqs)
        decode_waiting=len(self.decode_manager.running_reqs)
        with torch.cuda.nvtx.range('stage5_'+batch.phase):
            start.record(self.engine.stream)
            result=super()._forward(forward)
            end.record(self.engine.stream)
        self.gpu_batches.append((start,end,dict(phase=batch.phase,requests=batch.size,input_tokens=count,decode_waiting=decode_waiting)))
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--pools',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.job.read_text());job=source['job']
    rows=make_workload(json.loads(a.pools.read_text())[job['split']],job['scenario'],job['count'],job['rate'],job['seed'])
    assert workload_hash(rows)==source['workload_sha256']
    llm=ProfileLLM(a.model,cuda_graph_max_bs=8,cache_type='naive',num_page_override=65536,page_size=1,
                   max_running_req=8,max_seq_len_override=8192,max_extend_tokens=1024)
    llm.policy=job['policy'];llm.prefill_budget=1024;llm.deadline_s=120;llm.profiling=False
    try:
        llm.replay([dict(r,arrival=0,max_tokens=8) for r in rows[:8]],False)
        llm.profiling=True
        torch.cuda.cudart().cudaProfilerStart()
        try:result=llm.replay(rows,True)
        finally:torch.cuda.cudart().cudaProfilerStop()
        batches=[dict(**m,start_ms=llm.epoch.elapsed_time(start),end_ms=llm.epoch.elapsed_time(end)) for start,end,m in llm.gpu_batches]
        assert all(not v for v in llm.batch_policy.wait_since.values())
        (a.output/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in llm.events))
        (a.output/'outputs.json').write_text(json.dumps(llm.outputs)+'\n')
        record=dict(policy=job['policy'],scenario=job['scenario'],seed=job['seed'],count=len(rows),
            workload_sha256=workload_hash(rows),completed=len(llm.done),output_tokens=result['output_tokens'],
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            scheduler_sha256=hashlib.sha256(Path('python/minisgl/scheduler/scheduler.py').read_bytes()).hexdigest(),
            batches=batches,note='Real timed arrivals and policy selection under profiler. CUDA-event spans are not pure kernel durations. Not a replacement for five-repeat stage-four results.')
        (a.output/'gpu_timeline.json').write_text(json.dumps(record,indent=2)+'\n')
        print('PASS profile replay '+job['policy'],flush=True)
    finally:llm.shutdown()


if __name__=='__main__':main()
