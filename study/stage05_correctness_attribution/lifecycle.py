"""GPU lifecycle suite. Forced tokens isolate lifecycle from numerical generation differences."""
import argparse
import os
import time
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--cache',choices=('naive','radix'),required=True)
p.add_argument('--overlap',type=int,choices=(0,1),required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--waves',type=int,default=100)
p.add_argument('--soak-seconds',type=float,default=0)
a=p.parse_args()
os.environ['MINISGL_DISABLE_OVERLAP_SCHEDULING']=str(1-a.overlap)
import torch
from minisgl.llm.llm import LLM,RequestAllFinished
from minisgl.core import SamplingParams
from minisgl.message import UserMsg,AbortBackendMsg
from minisgl.scheduler.policy import BatchPolicy
from minisgl.scheduler.prefill import ChunkedReq

POLICIES=('prefill_first','decode_first','wait_time')


class LifecycleLLM(LLM):
    def offline_receive_msg(self,blocking=False):
        self.step+=1
        assert time.monotonic()-self.started < 90, 'Lifecycle case timed out'
        if self.step==0:
            return self.messages
        messages=[]
        if self.case=='cancel_pending_chunk':
            if self.step==1:
                assert any(r.uid==1 and r.chunked_req is None for r in self.prefill_manager.pending_list)
                self.aborted.add(1); messages.append(AbortBackendMsg(uid=1))
            if self.step==2:
                assert any(r.uid==0 and r.chunked_req is not None for r in self.prefill_manager.pending_list)
                self.aborted.add(0); messages.extend([AbortBackendMsg(uid=0),self.message(2,64,4,300)])
        if self.case=='cancel_decode' and self.step==1:
            self.aborted.update((0,1))
            messages=[AbortBackendMsg(uid=0),AbortBackendMsg(uid=1),self.message(2,64,4,400)]
        if messages:return messages
        if blocking:raise RequestAllFinished()
        return []

    def offline_send_result(self,reply):
        for r in reply:
            assert r.uid not in self.aborted, 'Output after cancellation'
            assert r.uid not in self.done, 'Duplicate output after terminal result'
            self.outputs.setdefault(r.uid,[]).append(r.next_token)
            if r.finished:self.done.add(r.uid)

    def _forward(self,forward):
        self.current_batch=forward.batch
        if forward.batch.is_prefill:
            self.prefill_chunks+=1
            self.cache_hits+=sum(r.cached_len>0 and not isinstance(r,ChunkedReq) for r in forward.batch.reqs)
        return super()._forward(forward)

    @staticmethod
    def message(uid,length,output,offset=100):
        return UserMsg(uid=uid,input_ids=torch.arange(offset,offset+length,dtype=torch.int32),
                       sampling_params=SamplingParams(temperature=0,max_tokens=output))

    def trial(self,case,messages,expected):
        self.case,self.messages=case,messages
        self.step=-1;self.outputs={};self.done=set();self.aborted=set();self.started=time.monotonic()
        self.prefill_chunks=0;self.cache_hits=0
        try:self.run_forever()
        except RequestAllFinished:pass
        torch.cuda.synchronize()
        assert self.done==set(expected),(case,self.done,expected)
        for uid,count in expected.items():
            assert len(self.outputs[uid])==count,(case,uid,len(self.outputs[uid]),count)
            assert all(t==(self.eos_token_id if case=='eos_length' and uid==0 else 100) for t in self.outputs[uid])
        self.cache_manager.check_integrity()
        assert self.table_manager.available_size==8
        assert len(set(self.table_manager._free_slots))==8
        assert not self.prefill_manager.pending_list and not self.decode_manager.running_reqs
        assert all(not x for x in self.batch_policy.wait_since.values())
        assert not getattr(self,'_inflight_reqs',{}) and not getattr(self,'_terminal_reqs',set())
        size=self.cache_manager.prefix_cache.size_info
        assert size.evictable_size==size.total_size
        return dict(case=case,completed=len(expected),chunks=self.prefill_chunks,cache_hits=self.cache_hits)


llm=LifecycleLLM(a.model,cuda_graph_max_bs=8,cache_type=a.cache,num_page_override=4096,
                 page_size=1,max_running_req=8,max_seq_len_override=2048,max_extend_tokens=256)
assert llm.eos_token_id!=100
original=llm.engine.sampler.sample

def sample(logits,args):
    result=original(logits,args)
    result.fill_(100)
    for i,r in enumerate(llm.current_batch.reqs):
        if llm.case=='eos_length' and r.uid==0:result[i]=llm.eos_token_id
    return result
llm.engine.sampler.sample=sample
original_evict=llm.cache_manager.prefix_cache.evict
llm.evicted_pages=0

def evict(count):
    result=original_evict(count)
    llm.evicted_pages+=len(result)
    return result
llm.cache_manager.prefix_cache.evict=evict
results=[]
a.output.parent.mkdir(parents=True,exist_ok=True)
started=time.monotonic()
try:
    for policy in POLICIES:
        llm.batch_policy=BatchPolicy(policy)
        for case in ('eos_length','length_cap','chunk_complete','cancel_pending_chunk','cancel_decode'):
            # Keep cancellation steps independent of prefixes from earlier cases.
            llm.cache_manager._free(llm.cache_manager.prefix_cache.evict(llm.cache_manager.prefix_cache.size_info.evictable_size))
            if case=='eos_length':messages=[llm.message(i,32,3) for i in range(2)];expected={0:1,1:3}
            elif case=='length_cap':messages=[llm.message(0,2040,20)];expected={0:8}
            elif case=='chunk_complete':messages=[llm.message(i,1024,10,100+i*2000) for i in range(2)];expected={0:10,1:10}
            elif case=='cancel_pending_chunk':messages=[llm.message(i,1024,10,6000+i*2000) for i in range(2)];expected={2:4}
            else:messages=[llm.message(i,32,10,100+i*2000) for i in range(2)];expected={2:4}
            results.append(dict(policy=policy,**llm.trial(case,messages,expected)))
        # Repeated prefix followed by distinct prefixes forces eviction in radix.
        for j in range(10):
            offset=12000 if j<2 else 12000+j*1100
            results.append(dict(policy=policy,**llm.trial('prefix_eviction',[llm.message(0,1024,4,offset)],{0:4})))
        hits=sum(r['cache_hits'] for r in results if r['policy']==policy and r['case']=='prefix_eviction')
        if a.cache=='radix':assert hits>0 and llm.evicted_pages>0
        for wave in range(a.waves):
            msgs=[llm.message(i,64+(wave%3)*128,8,30000+wave*8+i) for i in range(8)]
            llm.trial('recycle',msgs,{i:8 for i in range(8)})
        results.append(dict(policy=policy,case='recycle',waves=a.waves,completed=a.waves*8))
    soak_start=time.monotonic();soak_waves=0
    llm.batch_policy=BatchPolicy('wait_time')
    while time.monotonic()-soak_start<a.soak_seconds:
        wave=soak_waves
        llm.trial('soak',[llm.message(i,128+(wave%4)*128,16,60000+(wave%1000)*8+i) for i in range(8)],{i:16 for i in range(8)})
        soak_waves+=1
    report=dict(status='ok',cache=a.cache,overlap=bool(a.overlap),results=results,
        evicted_pages=llm.evicted_pages,seconds=time.monotonic()-started,
        soak_seconds=time.monotonic()-soak_start,soak_waves=soak_waves,soak_completed=soak_waves*8,
        forced_sampling=True,resource_checks_after_each_wave=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
except Exception as error:
    a.output.with_suffix('.failed.json').write_text(json.dumps(dict(status='failed',cache=a.cache,overlap=bool(a.overlap),case=llm.case,error_type=type(error).__name__,error=str(error)),indent=2)+'\n')
    raise
finally:llm.shutdown()
