"""Audit saved decisions independently of the policy implementation (CPU only)."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path


def expected_preference(decision):
    waits, thresholds = decision['wait_ms'], decision['thresholds_ms']
    overdue = {p: waits[p] is not None and waits[p] >= thresholds[p]
               for p in ('prefill', 'decode')}
    assert overdue == decision['overdue']
    if all(overdue.values()):
        preferred = ('decode' if waits['decode'] - thresholds['decode'] >=
                     waits['prefill'] - thresholds['prefill'] else 'prefill')
        return preferred, 'both_overdue'
    if overdue['decode']:
        return 'decode', 'decode_overdue'
    if overdue['prefill']:
        return 'prefill', 'prefill_overdue'
    return 'prefill', 'below_thresholds'


def audit(events):
    reasons, joint_selections = Counter(), Counter()
    pending_decision = None
    total = 0
    for event in events:
        if event['event'] == 'decision':
            assert pending_decision is None, 'Selected decision has no following batch'
            if event['policy'] == 'wait_time':
                assert expected_preference(event) == (event['preferred'], event['reason'])
                reasons[event['reason']] += 1
                if all(event['tracked_requests'][p] for p in ('prefill', 'decode')):
                    joint_selections[str(event['selected'])] += 1
            if event['selected'] is not None:
                assert event['fallback'] == (event['selected'] != event['preferred'])
                pending_decision = event
            else:
                assert event['fallback']
            total += 1
        elif event['event'] == 'batch':
            assert pending_decision is not None, 'Batch has no decision'
            assert event['phase'] == pending_decision['selected']
            if event['phase'] == 'prefill':
                assert event['input_tokens'] <= pending_decision['token_budget']
            pending_decision = None
    assert pending_decision is None
    assert total > 0
    return dict(decisions=total, reasons=dict(reasons),
                selections_with_both_phases_present=dict(joint_selections))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / 'summary.json').read_text())
    workloads = json.loads((args.directory / 'workloads.json').read_text())
    results = []
    for run in summary['runs']:
        name = f'{run["scenario"]}_{run["policy"]}_repeat{run["repetition"]}'
        events = [json.loads(line) for line in (args.directory / f'{name}.jsonl').read_text().splitlines()]
        result = audit(events)
        outputs = {int(k): v for k, v in json.loads(
            (args.directory / f'{name}_outputs.json').read_text()).items()}
        expected = {r['uid']: r['max_tokens'] for r in workloads[run['scenario']]}
        assert {uid: len(tokens) for uid, tokens in outputs.items()} == expected
        assert hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest() == run['sha256']
        if run['policy'] == 'wait_time' and run['scenario'] != 'short_only':
            counts = result['selections_with_both_phases_present']
            assert counts.get('prefill', 0) > 0 and counts.get('decode', 0) > 0
        results.append(dict(run=name, **result))
    comparisons = []
    for scenario in workloads:
        baseline = json.loads((args.directory / f'{scenario}_prefill_first_repeat0_outputs.json').read_text())
        for policy in summary['policies']:
            if policy == 'prefill_first':
                continue
            output = json.loads((args.directory / f'{scenario}_{policy}_repeat0_outputs.json').read_text())
            differences = []
            for uid, tokens in baseline.items():
                assert len(tokens) == len(output[uid])
                first = next((i for i, (a, b) in enumerate(zip(tokens, output[uid])) if a != b), None)
                if first is not None:
                    differences.append(dict(uid=int(uid), first_different_token_index=first,
                                            baseline_token=tokens[first], policy_token=output[uid][first]))
            comparisons.append(dict(scenario=scenario, policy=policy, repetition=0,
                                    different_requests=differences))
    (args.directory / 'output_comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (args.directory / 'decision_audit.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'PASS: decisions, fallback and prefill budget in {len(results)} runs')
    print('Both phases received service while both were present in every mixed wait_time run.')


if __name__ == '__main__':
    main()
