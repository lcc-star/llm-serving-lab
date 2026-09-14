"""Inventory first output divergences; publish counts, keep request identities local."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    groups = defaultdict(Counter)
    selected = None
    for path in sorted(a.source.glob('eval_*wait_time_b1024_n24_*.json')):
        if path.name.endswith('_outputs.json'):
            continue
        record = json.loads(path.read_text())
        baseline = path.stem.replace('wait_time', 'prefill_first')
        left = json.loads((a.source / (baseline + '_outputs.json')).read_text())
        right = json.loads(path.with_name(path.stem + '_outputs.json').read_text())
        counts = groups[record['job']['scenario']]
        targets = {}
        for uid in left:
            index = first_difference(left[uid], right[uid])
            counts['request_pairs'] += 1
            if index is not None:
                counts['different_pairs'] += 1
                counts['first_token' if index == 0 else 'later_token'] += 1
                targets[uid] = index
        if selected is None and record['job']['scenario'] == 'mixed_steady' and record['job']['level'] == 'near' and targets:
            selected = dict(baseline=baseline, alternative=path.stem, targets=targets)
    assert selected is not None
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'selected.json').write_text(json.dumps(selected, indent=2) + '\n')
    (a.output / 'inventory.json').write_text(json.dumps(dict(groups), indent=2) + '\n')
    print(json.dumps(dict(groups), indent=2))
    print('Selected one mixed_steady/near pair; request IDs remain local.')


if __name__ == '__main__':
    main()
