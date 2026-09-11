"""Validate saved causal controls; uses CPU only, no model download."""
import json
from pathlib import Path
import torch


def main():
    root=Path(__file__).parent/'evidence/divergence'
    data={name:json.loads((root/f'{name}.json').read_text()) for name in ('graph','eager','probe')}
    for name,runs in data.items():
        for i in range(0,len(runs),2):
            a,b=runs[i:i+2]
            assert a['schedule']==b['schedule']
            assert a['outputs']==b['outputs']
            assert a['observations']==b['observations']
    for a,b,c in zip(data['graph'],data['eager'],data['probe']):
        assert a['outputs']==b['outputs']==c['outputs']
        assert a['schedule']==b['schedule']==c['schedule']
    a,b=[next(r for r in data['probe'] if r['release_step']==s) for s in (11,12)]
    x,y=a['observations'][0],b['observations'][0]
    assert x['prefix_sha256']==y['prefix_sha256']
    assert x['batch_size']==y['batch_size']==8
    assert (x['argmax'],y['argmax'])==(3343,1210)
    assert a['outputs']['1'][:60]==b['outputs']['1'][:60]
    assert a['outputs']['1'][60]==3343 and b['outputs']['1'][60]==1210
    for obs in (x,y):
        scores=obs['high_precision_head_candidates']
        assert scores[1]>scores[0]
        rounded=torch.tensor(scores,dtype=torch.float64).to(torch.bfloat16).float().tolist()
        assert rounded==[obs['candidate_1210'],obs['candidate_3343']]
    scores=torch.full((3344,),-float('inf'))
    scores[1210]=scores[3343]=26
    assert scores.argmax().item()==1210
    print('PASS: repeatability, Graph/eager/probe agreement, same prefix, BF16 rounding and tie selection')


if __name__=='__main__':
    main()
