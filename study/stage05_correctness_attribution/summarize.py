"""Export anonymous numerical evidence; no vocabulary indices or reversible prefixes."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = {mode: json.loads((a.runs / mode / 'results.json').read_text()) for mode in ('graph', 'eager')}
    cases = []
    repeatability = {}
    recorded_match = {}
    for mode, records in data.items():
        assert len(records) == 4
        for name in ('baseline', 'alternative'):
            x, y = [r for r in records if r['case'] == name]
            assert x['repeat'] == 0 and y['repeat'] == 1
            repeatability[mode + '/' + name] = x['outputs'] == y['outputs'] and x['observations'] == y['observations']
            recorded_match[mode + '/' + name] = x['identical_to_recorded'] and y['identical_to_recorded']
        left = next(r for r in records if r['case'] == 'baseline')
        right = next(r for r in records if r['case'] == 'alternative')
        assert left['observations'].keys() == right['observations'].keys()
        for number, uid in enumerate(sorted(left['observations'], key=int)):
            x, y = left['observations'][uid], right['observations'][uid]
            same_prefix = x['prefix_sha256'] == y['prefix_sha256']
            index = x['index']; assert index == y['index']
            assert same_prefix == (left['outputs'][uid][:index] == right['outputs'][uid][:index])
            u = torch.load(a.runs / mode / f'baseline_r0_uid{uid}.pt', weights_only=True)
            v = torch.load(a.runs / mode / f'alternative_r0_uid{uid}.pt', weights_only=True)
            choices = [x['argmax'], y['argmax']]
            cases.append(dict(case_number=number, mode=mode, output_index=index, same_prefix=same_prefix, argmax_differs=x['argmax'] != y['argmax'],
                baseline_batch_size=x['batch_size'], alternative_batch_size=y['batch_size'],
                baseline_phase=x['phase'], alternative_phase=y['phase'], dtype=x['dtype'],
                baseline_top_two_margin=x['top_values'][0]-x['top_values'][1],
                alternative_top_two_margin=y['top_values'][0]-y['top_values'][1],
                baseline_candidate_scores=u[choices].tolist(), alternative_candidate_scores=v[choices].tolist(),
                max_absolute_logits_delta=(u-v).abs().max().item(),
                mean_absolute_logits_delta=(u-v).abs().mean().item()))
    agree = all(x['outputs'] == y['outputs'] and x['observations'] == y['observations']
                for x, y in zip(data['graph'], data['eager']))
    result = dict(runs=8, repeatability=repeatability, recorded_outputs_reproduced=recorded_match,
        graph_eager_outputs_equal=all(x['outputs']==y['outputs'] for x,y in zip(data['graph'],data['eager'])),
        graph_eager_observations_equal=agree, cases=cases,
        note='Same-prefix logits differences observed in one selected workload pair; does not identify the originating operator or rule out cache correctness defects.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'diagnostic_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'inventory.json').write_text((a.runs / 'inventory.json').read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
