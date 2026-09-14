"""Standalone latency-throughput figures from publishable aggregates."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'prefill_first':'#d88724','decode_first':'#3286bb','wait_time':'#7952b3'}
SCENARIOS=('short_only','long_only','mixed_steady','long_burst','decode_new','kv_tight')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    groups=json.loads((args.directory/'comparison.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,scenario in zip(axes.flat,SCENARIOS):
        for policy,color in COLORS.items():
            rows=[next((g for g in groups if g['split']=='eval' and g['scenario']==scenario and
                        g['policy']==policy and g['level']==level and g['budget']==1024),None)
                  for level in ('low','medium','near')]
            rows=[g for g in rows if g is not None]
            if not rows:continue
            key='long_itl_p99_ms' if scenario=='long_only' else 'short_itl_p99_ms'
            x=[g['metrics']['output_tokens_per_s']['median'] for g in rows]
            y=[g['metrics'][key]['median'] for g in rows]
            ax.plot(x,y,'o-',color=color,label=policy)
            for a,b,g in zip(x,y,rows):
                v=g['metrics'][key]
                ax.errorbar(a,b,yerr=[[b-v['min']],[v['max']-b]],color=color,capsize=3)
                ax.annotate(g['level'],(a,b),xytext=(4,4),textcoords='offset points',fontsize=7)
        ax.set_title(scenario)
        ax.set_xlabel('Output tokens / second')
        ax.set_ylabel(('Long' if scenario=='long_only' else 'Short')+' request P99 ITL (ms)')
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Stage 4: CPU-observed latency vs throughput; median and min–max over five runs')
    fig.savefig(args.directory/'latency_throughput.svg')
    fig.savefig(args.directory/'latency_throughput.png',dpi=150)
    plt.close(fig)
    confirm=[g for g in groups if g['level']=='confirm']
    if confirm:
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,scenario in zip(axes,('mixed_steady','long_burst')):
            rows=[g for g in confirm if g['scenario']==scenario]
            labels=[f"{g['policy']}\nbudget={g['budget']}" for g in rows]
            values=[g['metrics']['short_itl_p99_ms'] for g in rows]
            ax.bar(labels,[v['median'] for v in values],color=[COLORS[g['policy']] for g in rows],
                   yerr=[[v['median']-v['min'] for v in values],[v['max']-v['median'] for v in values]],capsize=4)
            ax.set_title(scenario+' / 80 requests per run')
            ax.set_ylabel('Short request P99 ITL (ms)')
            ax.tick_params(axis='x',labelsize=8)
            ax.grid(axis='y',alpha=.25)
        fig.savefig(args.directory/'confirmation.svg')
        fig.savefig(args.directory/'confirmation.png',dpi=150)
        plt.close(fig)
    print('Figures exported; matplotlib '+matplotlib.__version__)


if __name__=='__main__':main()
