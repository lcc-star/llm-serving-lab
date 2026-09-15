"""Stage-six entry point; reuse experiments and keep new raw artifacts local."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'study'


def evidence():
    rows = json.loads((STUDY / 'stage04_system_evaluation/evidence/comparison.json').read_text())
    for scenario, budget in [('mixed_steady', 1024), ('long_burst', 4096)]:
        def select(policy, selected_budget):
            matches = [r for r in rows if r['scenario'] == scenario and r['level'] == 'confirm'
                       and r['policy'] == policy and r['budget'] == selected_budget]
            assert len(matches) == 1
            row = matches[0]
            assert row['count'] == 80 and row['repeats'] == 5
            return row['metrics']
        baseline, candidate = select('prefill_first', 1024), select('wait_time', budget)
        old, new = [m['short_itl_p99_ms']['median'] for m in (baseline, candidate)]
        throughput = (candidate['output_tokens_per_s']['median'] /
                      baseline['output_tokens_per_s']['median'] - 1) * 100
        print(f'{scenario}: short P99 {old:.2f} -> {new:.2f} ms; '
              f'reduction {(1-new/old)*100:.2f}%; throughput {throughput:+.2f}%')
    result = json.loads((STUDY / 'stage05_correctness_attribution/evidence/lifecycle_summary.json').read_text())
    assert result['status'] == 'passed'
    print(f"Stage-five tests: {result['cpu_passed']}; soak requests: {result['soak_completed']}")
    print('Published aggregates verified; no GPU measurements were executed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['evidence', 'cpu', 'stage3', 'evaluation', 'lifecycle'])
    parser.add_argument('--model', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.mode == 'evidence':
        evidence()
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = STUDY / 'stage06_interview/runs' / (args.mode + '_' + stamp)
    python = sys.executable
    commands = []
    if args.mode == 'cpu':
        commands = [[python, '-m', 'unittest', 'discover', '-s', 'tests/' + suite, '-v']
                    for suite in ['stage01_measurement', 'stage02', 'stage03', 'stage04', 'stage05']]
    else:
        if args.model is None:
            parser.error('--model is required for GPU experiments')
        model = str(args.model.expanduser().resolve())
        if not args.dry_run and not Path(model).is_dir():
            parser.error('--model must be an existing local model directory')
        if args.mode == 'stage3':
            commands = [[python, 'study/stage03_scheduling/run.py', '--model', model,
                         '--output', str(out), '--policies', 'prefill_first', 'decode_first', 'wait_time',
                         '--decode-wait-ms', '50', '--prefill-wait-ms', '200'],
                        [python, 'study/stage03_scheduling/summarize.py', str(out)],
                        [python, 'study/stage03_scheduling/audit_decisions.py', str(out)]]
        elif args.mode == 'evaluation':
            if args.dataset is None:
                parser.error('--dataset is required for evaluation')
            dataset = str(args.dataset.expanduser().resolve())
            if not args.dry_run and not Path(dataset).is_file():
                parser.error('--dataset must be an existing ShareGPT file')
            prefix = 'study/stage04_system_evaluation/'
            prepared, measured = out / 'prepared', out / 'measurement'
            commands = [[python, prefix + 'prepare.py', '--dataset', dataset, '--model', model,
                         '--output', str(prepared)],
                        [python, prefix + 'run.py', '--model', model, '--pools', str(prepared / 'pools.json'),
                         '--output', str(measured), '--stage', 'all', '--repeats', '5'],
                        [python, prefix + 'summarize.py', '--runs', str(measured),
                         '--pools', str(prepared / 'pools.json'), '--output', str(out / 'evidence')],
                        [python, prefix + 'plot.py', str(out / 'evidence')]]
        else:
            prefix = 'study/stage05_correctness_attribution/'
            commands = [[python, prefix + 'run_lifecycle.py', '--model', model,
                         '--output', str(out / 'lifecycle'), '--waves', '100', '--soak-seconds', '600']]
            commands += [[python, prefix + 'extra_lifecycle.py', '--model', model,
                          '--cache', cache, '--overlap', str(overlap), '--output',
                          str(out / 'extra' / f'{cache}_overlap{overlap}.json')]
                         for cache in ['naive', 'radix'] for overlap in [0, 1]]
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + env.get('PYTHONPATH', '')
    # Existing stage-four runner selects physical GPU 0; use the same documented device.
    env['CUDA_VISIBLE_DEVICES'] = '0'
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                      mode=args.mode, commands=commands)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for index, command in enumerate(commands):
        print(f'Running step {index + 1}/{len(commands)}; logs: {out}', flush=True)
        with (out / f'step_{index + 1}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'PASS {args.mode}; local artifacts: {out}')


if __name__ == '__main__':
    main()
