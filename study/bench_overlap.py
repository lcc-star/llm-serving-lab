"""Controlled single-GPU overlap scheduling experiment; run each mode in a new process."""
import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault('MINISGL_DISABLE_OVERLAP_SCHEDULING', '1')

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.env import ENV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=int, choices=[0, 8], required=True)
    parser.add_argument('--requests', type=int, choices=[1, 8], default=8)
    args = parser.parse_args()
    rng = random.Random(0)
    prompts = [[rng.randrange(100, 10000) for _ in range(128)] for _ in range(args.requests)]
    llm = LLM(
        os.environ.get('MODEL_PATH', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
        cuda_graph_max_bs=args.graph, cache_type='naive',
        num_page_override=4096, page_size=1, max_running_req=8,
        max_seq_len_override=512, max_extend_tokens=1024,
    )
    runs = []
    try:
        for iteration in range(6):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, SamplingParams(temperature=0, ignore_eos=True, max_tokens=128))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ids = [result['token_ids'] for result in outputs]
            assert len(ids) == args.requests and all(len(tokens) == 128 for tokens in ids), [len(t) for t in ids]
            record = dict(seconds=elapsed, output_tokens=args.requests * 128, tokens_per_second=args.requests * 128 / elapsed,
                          output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            print(json.dumps(dict(iteration=iteration, warmup=iteration == 0, **record)), flush=True)
            if iteration:
                runs.append(record)
        result = dict(graph_max_bs=args.graph, overlap=not bool(ENV.DISABLE_OVERLAP_SCHEDULING), cache='naive', requests=args.requests,
                      input_tokens_per_request=128, output_tokens_per_request=128,
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(), runs=runs,
                      median_seconds=statistics.median(r['seconds'] for r in runs),
                      median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in runs))
        suffix = '' if args.requests == 8 else f'_requests{args.requests}'
        Path(__file__).with_name(f'overlap_{int(not bool(ENV.DISABLE_OVERLAP_SCHEDULING))}_graph{args.graph}{suffix}.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        llm.shutdown()


if __name__ == '__main__':
    main()
