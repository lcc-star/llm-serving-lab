"""Deterministic open-loop arrivals over private, conversation-disjoint pools."""
import hashlib
import json
import math
import random

SCENARIOS = ('short_only', 'long_only', 'mixed_steady', 'long_burst', 'decode_new', 'kv_tight')
POLICIES = ('prefill_first', 'decode_first', 'wait_time')


def make_workload(pool, scenario, count, rate, seed):
    if scenario not in SCENARIOS or count < 8 or count % 2 or not math.isfinite(rate) or rate < 0:
        raise ValueError('Expected a known scenario, even count >= 8 and nonnegative rate')
    prompt_rng = random.Random(seed)
    arrival_rng = random.Random(seed + 1_000_003)
    if scenario == 'short_only':
        kinds = ['short'] * count
    elif scenario == 'long_only':
        kinds = ['long'] * count
    elif scenario == 'decode_new':
        kinds = ['short'] * 4 + ['long'] * (count - 4)
    else:
        kinds = ['short'] * (count // 2) + ['long'] * (count // 2)
        if scenario in ('mixed_steady', 'kv_tight'):
            prompt_rng.shuffle(kinds)
    samples = {kind: iter(prompt_rng.sample([r for r in pool if r['bucket'] == kind],
                                           kinds.count(kind))) for kind in sorted(set(kinds))}
    rows, t, short_t = [], 0.0, 0.0
    for uid, kind in enumerate(kinds):
        if rate == 0:
            arrival = 0.0  # Saturation pilot only.
        elif scenario == 'long_burst':
            if kind == 'short':
                if uid:
                    short_t += arrival_rng.expovariate(rate)
                arrival = short_t
            else:
                arrival = .3 + ((uid - count // 2) // 4) * 4 / rate
        elif scenario == 'decode_new':
            if uid < 4:
                arrival = 0.0
            else:
                t += arrival_rng.expovariate(rate)
                arrival = .3 + t
        else:
            if uid:
                t += arrival_rng.expovariate(rate)
            arrival = t
        sample = next(samples[kind])
        rows.append(dict(uid=uid, arrival=arrival, input_ids=sample['input_ids'],
                         max_tokens=(512 if scenario == 'decode_new' and kind == 'short'
                                     else 256 if kind == 'short' else 128), kind=kind))
    return sorted(rows, key=lambda r: (r['arrival'], r['uid']))


def workload_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def page_capacity(scenario):
    return 4608 if scenario == 'kv_tight' else 65536
