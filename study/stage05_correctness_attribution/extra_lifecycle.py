"""Explicit shared-prefix and already-launched decode cancellation checks."""
import argparse
import os
import json
import hashlib
import subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--cache',choices=('naive','radix'),required=True);p.add_argument('--overlap',type=int,choices=(0,1),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ['MINISGL_DISABLE_OVERLAP_SCHEDULING']=str(1-a.overlap)
import torch
from minisgl.llm.llm import LLM,RequestAllFinished
from minisgl.message import UserMsg,AbortBackendMsg
from minisgl.core import SamplingParams
from minisgl.scheduler.policy import BatchPolicy


class ExtraLLM(LLM):
    def msg(self,uid,n,out,offset=100):
        return UserMsg(uid=uid,input_ids=torch.arange(offset,offset+n,dtype=torch.int32),sampling_params=SamplingParams(max_tokens=out))
    def offline_receive_msg(self,blocking=False):
        self.step+=1
        assert self.step<1000
        if self.step==0:return self.messages
        if self.case=='cancel_inflight_decode' and self.step==2:
            assert self.decode_launched
            self.aborted={0,1}
            return [AbortBackendMsg(uid=0),AbortBackendMsg(uid=1),self.msg(2,32,4,400)]
        if blocking:raise RequestAllFinished()
        return []
    def offline_send_result(self,reply):
        for r in reply:
            assert r.uid not in self.aborted and r.uid not in self.done
            self.outputs.setdefault(r.uid,[]).append(r.next_token)
            if r.finished:self.done.add(r.uid)
    def _forward(self,forward):
        b=forward.batch;self.batch=b
        if a.cache=='radix':
            for req in b.reqs:
                cached=req.cache_handle.cached_len
                if cached:
                    assert torch.equal(self.engine.page_table[req.table_idx,:cached],req.cache_handle.get_matched_indices()), 'Live prefix still references duplicate pages'
                live=self.engine.page_table[req.table_idx,:req.device_len].cpu().tolist()
                assert not set(live).intersection(self.cache_manager.free_slots.cpu().tolist())
        if b.is_decode:self.decode_launched=True
        if self.case=='shared' and b.is_prefill and a.cache=='radix':
            assert len(b.reqs)==2
            first,second=b.reqs
            x,y=first.cache_handle.get_matched_indices(),second.cache_handle.get_matched_indices()
            assert first.cached_len==second.cached_len==1023
            assert torch.equal(x,y)
            assert self.cache_manager.prefix_cache.size_info.protected_size>=1023
            self.shared_verified=True
        return super()._forward(forward)
    def trial(self,case,messages,expected):
        self.case=case;self.messages=messages;self.step=-1;self.aborted=set();self.done=set();self.outputs={};self.counts={};self.decode_launched=False;self.shared_verified=False
        try:self.run_forever()
        except RequestAllFinished:pass
        torch.cuda.synchronize()
        assert self.done==set(expected)
        assert all(len(self.outputs[u])==n for u,n in expected.items())
        if case=='late_eos':assert self.outputs[0]==[100,100,self.eos_token_id]
        elif case=='shared':assert all(v==[100+u]*len(v) for u,v in self.outputs.items())
        else:assert all(all(t==100 for t in v) for v in self.outputs.values())
        self.cache_manager.check_integrity()
        assert self.table_manager.available_size==8 and len(set(self.table_manager._free_slots))==8
        assert not self._inflight_reqs and not self._terminal_reqs
        assert not self.prefill_manager.pending_list and not self.decode_manager.running_reqs
        assert all(not v for v in self.batch_policy.wait_since.values())
        sizes=self.cache_manager.prefix_cache.size_info
        assert sizes.protected_size==0
        if case=='shared' and a.cache=='radix':assert self.shared_verified
        return dict(case=case,completed=len(expected),shared_verified=self.shared_verified,decode_launched=self.decode_launched)


llm=ExtraLLM(a.model,cuda_graph_max_bs=8,cache_type=a.cache,num_page_override=4096,page_size=1,max_running_req=8,max_seq_len_override=2048,max_extend_tokens=256)
original=llm.engine.sampler.sample

def sample(logits,args):
    result=original(logits,args);result.fill_(100)
    for i,r in enumerate(llm.batch.reqs):
        if llm.case=='shared' and r.uid==1:result[i]=101
        llm.counts[r.uid]=llm.counts.get(r.uid,0)+1
        if llm.case=='late_eos' and r.uid==0 and llm.counts[r.uid]==3:result[i]=llm.eos_token_id
    return result
llm.engine.sampler.sample=sample
results=[]
try:
    for policy in ('prefill_first','decode_first','wait_time'):
        llm.batch_policy=BatchPolicy(policy)
        llm.trial('warm',[llm.msg(0,1024,4)],{0:4})
        results.append(dict(policy=policy,**llm.trial('shared',[llm.msg(i,1024,8) for i in range(2)],{0:8,1:8})))
        results.append(dict(policy=policy,**llm.trial('cancel_inflight_decode',[llm.msg(i,64,8,8000+i*100) for i in range(2)],{2:4})))
        results.append(dict(policy=policy,**llm.trial('late_eos',[llm.msg(i,64,8,9000+i*100) for i in range(2)],{0:3,1:8})))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    sources=['python/minisgl/scheduler/scheduler.py','python/minisgl/scheduler/cache.py',str(Path(__file__).resolve().relative_to(root))]
    provenance=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})
    a.output.write_text(json.dumps(dict(status='ok',cache=a.cache,overlap=bool(a.overlap),results=results,forced_sampling=True,provenance=provenance),indent=2)+'\n')
    print('PASS extra '+a.cache+' overlap='+str(a.overlap),flush=True)
finally:llm.shutdown()
