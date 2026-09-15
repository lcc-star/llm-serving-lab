"""Plot aggregate layer differences; no request tensors are read."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('evidence',type=Path)
    a=p.parse_args()
    data=json.loads((a.evidence/'hidden_summary.json').read_text())['decode']
    layers=list(range(32))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    ax.plot(layers,[data[f'layer{i}/attention']['max_abs'] for i in layers],'o-',label='Attention output')
    ax.plot(layers,[data[f'layer{i}/history_k']['max_abs'] for i in layers],'s--',label='Historical K')
    ax.plot(layers,[data[f'layer{i}/history_v']['max_abs'] for i in layers],'x:',label='Historical V')
    ax.set(xlabel='Layer (zero-based)',ylabel='Maximum absolute difference',
           title='Same-prefix decode: one selected request, two recorded batch schedules')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(a.evidence/'hidden_differences.png',dpi=150)
    fig.savefig(a.evidence/'hidden_differences.svg')
    f=a.evidence/'hidden_differences.svg'
    f.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
