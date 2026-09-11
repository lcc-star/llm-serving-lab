"""Metrics use scheduler CPU observation times, never GPU completion timestamps."""
import math


def percentiles(values):
    if not values:
        return {'count': 0, 'p50': None, 'p95': None, 'p99': None, 'max': None}
    values = sorted(values)
    def quantile(q):
        index = (len(values) - 1) * q
        low = math.floor(index)
        return values[low] + (values[math.ceil(index)] - values[low]) * (index - low)
    return dict(count=len(values), p50=quantile(.5), p95=quantile(.95),
                p99=quantile(.99), max=values[-1])


def summarize(events):
    requests = {e['uid']: e for e in events if e['event'] == 'request'}
    tokens = {uid: [] for uid in requests}
    first = {}
    for e in events:
        if e['event'] == 'batch':
            for uid in e['uids']:
                first.setdefault(uid, e['t'])
        elif e['event'] == 'token':
            tokens[e['uid']].append(e)
    ttft, e2e, itls, waits, ingress = [], [], [], [], []
    complete = 0
    for uid, req in requests.items():
        ts = tokens[uid]
        assert req['received'] >= req['arrival'] >= 0
        ingress.append(req['received'] - req['arrival'])
        if uid in first:
            assert first[uid] >= req['received']
            waits.append(first[uid] - req['arrival'])
        if not ts:
            continue
        assert ts[0]['t'] >= first[uid]
        assert all(b['t'] >= a['t'] for a, b in zip(ts, ts[1:]))
        assert not any(t['finished'] for t in ts[:-1]), 'Tokens after completion'
        ttft.append(ts[0]['t'] - req['arrival'])
        itls.extend(b['t'] - a['t'] for a, b in zip(ts, ts[1:]))
        if ts[-1]['finished']:
            complete += 1
            e2e.append(ts[-1]['t'] - req['arrival'])
    all_tokens = [t for ts in tokens.values() for t in ts]
    duration = (max(t['t'] for t in all_tokens) - min(r['arrival'] for r in requests.values())) if all_tokens else 0
    return dict(requests=len(requests), completed=complete, incomplete=len(requests)-complete,
                output_tokens=len(all_tokens), duration_s=duration,
                output_tokens_per_s=len(all_tokens)/duration if duration else 0,
                ttft_s=percentiles(ttft), itl_s=percentiles(itls), e2e_s=percentiles(e2e),
                arrival_to_first_schedule_s=percentiles(waits), ingress_lag_s=percentiles(ingress),
                timestamp_domain='CPU scheduler observation, not network or GPU time',
                p99_note='Descriptive only; small samples do not support reliable tail claims.')
