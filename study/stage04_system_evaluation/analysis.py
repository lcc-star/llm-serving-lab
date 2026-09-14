"""Aggregate metrics only: no prompt text, tokens, source IDs or local paths."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage01_measurement'))
from metrics import summarize, percentiles


def analyze(events, rows):
    result = summarize(events)
    groups = {r['uid']: r['kind'] for r in rows}
    starts = {r['uid']: r['arrival'] for r in rows}
    tokens = {r['uid']: [] for r in rows}
    first, last_decode, decode_intervals = {}, {}, []
    batches = [e for e in events if e['event'] == 'batch']
    for e in events:
        if e['event'] == 'token':
            tokens[e['uid']].append(e['t'])
        if e['event'] == 'batch':
            for uid in e['uids']:
                first.setdefault(uid, e['t'])
                if e['phase'] == 'decode':
                    if uid in last_decode:
                        decode_intervals.append(e['t'] - last_decode[uid])
                    last_decode[uid] = e['t']
    for kind in ('short', 'long'):
        selected = [u for u in tokens if groups[u] == kind]
        gaps = [b-a for u in selected for a,b in zip(tokens[u], tokens[u][1:])]
        result[f'{kind}_itl_s'] = percentiles(gaps)
        result[f'{kind}_ttft_s'] = percentiles([tokens[u][0]-starts[u] for u in selected if tokens[u]])
    last_arrival = max(starts.values())
    outstanding = [sum(r['arrival'] <= t and (not tokens[r['uid']] or tokens[r['uid']][-1] > t)
                       for r in rows) for t in (last_arrival,)]
    result.update(error_rate=result['incomplete']/len(rows),
                  request_count=len(rows), errors=result['incomplete'],
                  max_pending=max((b['pending'] for b in batches), default=0),
                  max_decode_running=max((b['decode_running'] for b in batches), default=0),
                  min_kv_free_pages=min((b['kv_free_pages'] for b in batches), default=0),
                  peak_kv_allocated_pages=max((b['kv_allocated_pages'] for b in batches), default=0),
                  outstanding_at_last_arrival=outstanding[0],
                  drain_seconds=max(0, max((t[-1] for t in tokens.values() if t), default=0)-last_arrival),
                  decode_schedule_interval_s=percentiles(decode_intervals),
                  max_first_schedule_wait_s=max((first[u]-starts[u] for u in first), default=0),
                  input_lengths=percentiles([len(r['input_ids']) for r in rows]),
                  output_lengths=percentiles([r['max_tokens'] for r in rows]))
    return result
