"""Recompute metrics and export a standalone SVG CPU-observation timeline."""
import argparse
import html
import json
from pathlib import Path
from metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('events', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    result = summarize(events)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    reqs = [e for e in events if e['event']=='request']
    duration = max((e.get('t',0) for e in events), default=1) or 1
    scale = lambda t: 150 + 1000*t/duration
    ymap = {r['uid']: 80+i*32 for i,r in enumerate(reqs)}
    height = 150+32*len(reqs)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="{height}" viewBox="0 0 1250 {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="13">',
           '<text x="20" y="25">CPU timeline: gray=arrival, orange=prefill scheduling, blue=decode scheduling, green=token observed</text>',
           '<text x="20" y="45">Markers indicate CPU events, not GPU execution duration. One row per request.</text>']
    for req in reqs:
        y=ymap[req['uid']]
        svg.append(f'<text x="15" y="{y+4}">request {html.escape(str(req["uid"]))}</text>')
        svg.append(f'<line x1="150" x2="1150" y1="{y}" y2="{y}" stroke="#ddd"/>')
        x=scale(req['arrival'])
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="gray"/>')
    for event in events:
        if event['event']=='batch':
            color = '#db7900' if event['phase']=='prefill' else '#2674d9'
            x=scale(event['t'])
            for uid in event['uids']:
                y=ymap[uid]
                svg.append(f'<line x1="{x}" x2="{x}" y1="{y-9}" y2="{y}" stroke="{color}"/>')
        elif event['event']=='token':
            svg.append(f'<circle cx="{scale(event["t"])}" cy="{ymap[event["uid"]]+5}" r="2" fill="#218838"/>')
    for i in range(6):
        t=duration*i/5
        svg.append(f'<text x="{scale(t)}" y="{height-20}">{t:.3f}s</text>')
    svg.append('</g></svg>')
    (args.output/'timeline.svg').write_text('\n'.join(svg))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
