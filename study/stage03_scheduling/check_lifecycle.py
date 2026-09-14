"""Real GPU model execution with deterministic sampled tokens to test scheduler lifecycle."""
import argparse
import json
import os
from pathlib import Path
os.environ['MINISGL_DISABLE_OVERLAP_SCHEDULING']='1'
import torch
from minisgl.llm.llm import LLM,RequestAllFinished
from minisgl.message import UserMsg,AbortBackendMsg
from minisgl.core import SamplingParams
from minisgl.scheduler.policy import BatchPolicy,SUPPORTED_POLICIES


class CheckLLM(LLM):
    def offline_receive_msg(self,blocking=False):
        self.step+=1
        if self.step==0:
            length=1024 if self.case=='cancel_pending_and_chunk' else 32
            return [UserMsg(uid=i,input_ids=torch.arange(100,100+length,dtype=torch.int32),
                            sampling_params=SamplingParams(max_tokens=3 if self.case=='eos_and_length' else 10)) for i in range(2)]
        if self.case=='cancel_pending_and_chunk':
            if self.step==1:
                assert self.prefill_manager.pending_list[0].chunked_req is not None
                assert self.prefill_manager.pending_list[1].chunked_req is None
                return [AbortBackendMsg(uid=1)]
            if self.step==2:return [AbortBackendMsg(uid=0)]
        if self.case=='cancel_decode' and self.step==1:
            assert len(self.decode_manager.running_reqs)==2
            return [AbortBackendMsg(uid=0),AbortBackendMsg(uid=1)]
        if blocking:
            raise RequestAllFinished()
        return []

    def offline_send_result(self,reply):
        self.replies.extend(dict(uid=m.uid,token=m.next_token,finished=m.finished) for m in reply)

    def _forward(self,forward):
        self.current_batch=forward.batch
        return super()._forward(forward)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    llm=CheckLLM(a.model,cuda_graph_max_bs=8,cache_type='naive',num_page_override=4096,page_size=1,
                max_running_req=8,max_seq_len_override=2048,max_extend_tokens=256)
    original=llm.engine.sampler.sample
    def sample(logits,args):
        result=original(logits,args)
        result.fill_(100) # guaranteed non-EOS for this Llama model, asserted below
        for i,r in enumerate(llm.current_batch.reqs):
            if llm.case=='eos_and_length' and r.uid==0:result[i]=llm.eos_token_id
        return result
    assert llm.eos_token_id!=100
    llm.engine.sampler.sample=sample
    results=[]
    try:
        for policy in SUPPORTED_POLICIES:
            for case in ('eos_and_length','cancel_pending_and_chunk','cancel_decode'):
                llm.batch_policy=BatchPolicy(policy);llm.case=case;llm.step=-1;llm.replies=[]
                try:llm.run_forever()
                except RequestAllFinished:pass
                torch.cuda.synchronize()
                llm.cache_manager.check_integrity()
                assert llm.table_manager.available_size==8
                assert all(not waits for waits in llm.batch_policy.wait_since.values())
                assert not llm.prefill_manager.pending_list and not llm.decode_manager.running_reqs
                if case=='eos_and_length':
                    r0=[r for r in llm.replies if r['uid']==0];r1=[r for r in llm.replies if r['uid']==1]
                    assert len(r0)==1 and r0[0]['finished'] and r0[0]['token']==llm.eos_token_id
                    assert len(r1)==3 and [r['finished'] for r in r1]==[False,False,True]
                elif case=='cancel_pending_and_chunk':assert not llm.replies
                else:assert len(llm.replies)==2 and not any(r['finished'] for r in llm.replies)
                results.append(dict(policy=policy,case=case,passed=True,replies=llm.replies))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(results,indent=2)+'\n')
        print(f'PASS: {len(results)} GPU lifecycle cases (EOS, length, waiting/chunk/decode cancellation)')
    finally:llm.shutdown()


if __name__=='__main__':main()
