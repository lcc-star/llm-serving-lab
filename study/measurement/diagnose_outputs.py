"""Fixed-step batch interventions; CPU logits copies make this unsuitable for timing."""
import argparse
import hashlib
import json
from pathlib import Path
from replay import ReplayLLM, make_rows, torch, UserMsg, SamplingParams, RequestAllFinished


class DiagnosticLLM(ReplayLLM):
    def reset_replay(self, rows, trace):
        super().reset_replay(rows, trace)
        self.step = -1
        self.observations = []
        self.schedule = []

    def offline_receive_msg(self, blocking=False):
        self.step += 1
        if blocking and self.cursor == len(self.rows):
            assert len(self.done) == len(self.rows)
            raise RequestAllFinished()
        messages = []
        while self.cursor < len(self.rows):
            row = self.rows[self.cursor]
            release = 0 if row['uid'] < 4 else self.release_step + row['uid'] - 4
            if release > self.step:
                break
            self.cursor += 1
            messages.append(UserMsg(uid=row['uid'], input_ids=self.inputs[row['uid']],
                sampling_params=SamplingParams(temperature=0,ignore_eos=True,max_tokens=row['max_tokens'])))
        return messages

    def _forward(self, forward):
        self.current_batch = forward.batch
        self.schedule.append(dict(step=self.step,phase=forward.batch.phase,
                                  uids=[r.uid for r in forward.batch.reqs],
                                  cached=[r.cached_len for r in forward.batch.reqs]))
        return super()._forward(forward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--graph',type=int,choices=[0,8],default=8)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--head-fp32',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert not args.head_fp32 or args.graph == 0, 'FP32 head diagnostic requires eager mode'
    rows=make_rows()
    llm=DiagnosticLLM(args.model,cuda_graph_max_bs=args.graph,cache_type='naive',num_page_override=8192,
                     page_size=1,max_running_req=8,max_seq_len_override=1024,max_extend_tokens=256)
    if args.head_fp32:
        head = llm.engine.model.lm_head
        head.weight = head.weight.float()
        head_forward = head.forward
        head.forward = lambda x: head_forward(x.float())
    # Counterfactual dot products on the exact hidden row and stored head weights.
    # Only two candidates, CPU float64; normal model outputs are untouched.
    head = llm.engine.model.lm_head
    candidate_weights = head.weight[[1210,3343]].double().cpu()
    head_forward_probe = head.forward
    def probe_head(x):
        for i, req in enumerate(llm.current_batch.reqs):
            if req.uid == 1 and len(llm.outputs[1]) == 60:
                llm.head_probe = torch.mv(candidate_weights, x[i].double().cpu()).tolist()
        return head_forward_probe(x)
    if args.graph == 0:
        head.forward = probe_head
    original=llm.engine.sampler.sample
    def sample(logits, sampling_args):
        batch=llm.current_batch
        for i,req in enumerate(batch.reqs):
            if req.uid == 1 and len(llm.outputs[1]) == 60:
                values=logits[i].float().cpu()
                top=torch.topk(values,5)
                prefix=llm.rows[1]['input_ids']+llm.outputs[1]
                llm.observations.append(dict(step=llm.step,batch_size=batch.size,
                    padded_size=batch.padded_size,dtype=str(logits.dtype),
                    prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                    top_ids=top.indices.tolist(),top_values=top.values.tolist(),
                    candidate_1210=values[1210].item(),candidate_3343=values[3343].item(),
                    argmax=int(values.argmax()),
                    high_precision_head_candidates=getattr(llm,'head_probe',None),
                    logits_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest()))
                torch.save(values,args.output/f'logits_step{llm.release_step}_repeat{llm.repeat}.pt')
        return original(logits,sampling_args)
    llm.engine.sampler.sample=sample
    results=[]
    try:
        for release in ([11,12] if args.head_fp32 else [2,6,10,11,12,14,20]):
            for repeat in range(2):
                llm.release_step=release
                llm.repeat=repeat
                result=llm.replay(rows,False)
                record=dict(release_step=release,repeat=repeat,graph=args.graph,head_fp32=args.head_fp32,
                            outputs=llm.outputs,observations=llm.observations,schedule=llm.schedule,**result)
                results.append(record)
                print(json.dumps({k:record[k] for k in ['release_step','repeat','observations']}),flush=True)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:
        llm.shutdown()


if __name__=='__main__':
    main()
