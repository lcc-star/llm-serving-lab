"""Prepare private tokenized ShareGPT pools and a publishable aggregate manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

EXPECTED_SHA256 = '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def first_prompt(record):
    """Use an intact opening user turn, optionally preceded by one system turn."""
    turns = record.get('conversations', [])
    if not turns:
        return None
    messages = []
    if turns[0].get('from') == 'system':
        messages.append({'role': 'system', 'content': turns[0].get('value')})
        turns = turns[1:]
    if not turns or turns[0].get('from') not in ('human', 'user'):
        return None
    messages.append({'role': 'user', 'content': turns[0].get('value')})
    if any(not isinstance(m['content'], str) or not m['content'].strip() for m in messages):
        return None
    return messages


def group_id(record):
    return re.sub(r'_\d+$', '', str(record['id']))


def split_for_group(group):
    # Conversation-level, independent of sampling/arrival seeds.
    h = hashlib.sha256(('stage04-split-v1:' + group).encode()).digest()
    return 'tune' if int.from_bytes(h[:8], 'big') % 5 == 0 else 'eval'


def bucket(length):
    return 'short' if length <= 512 else 'long' if length >= 2048 else 'medium'


def length_stats(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return dict(count=len(values), min=values[0], max=values[-1],
                p50=values[(len(values)-1)//2], p95=values[int((len(values)-1)*.95)],
                p99=values[int((len(values)-1)*.99)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = file_hash(args.dataset)
    if digest != EXPECTED_SHA256:
        raise ValueError('Dataset SHA256 differs from the reviewed file')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = json.loads(args.dataset.read_text())
    source_records = len(records)
    skipped, seen, candidates = Counter(), set(), []
    for record in records:
        messages = first_prompt(record)
        if messages is None:
            skipped['invalid_or_continuation_opening'] += 1
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_hash = hashlib.sha256(text.encode()).hexdigest()
        if prompt_hash in seen:
            skipped['duplicate_rendered_prompt'] += 1
            continue
        seen.add(prompt_hash)
        candidates.append((group_id(record), prompt_hash, text))
    del records
    pools = {'tune': [], 'eval': []}
    all_lengths = []
    for start in range(0, len(candidates), 256):
        batch = candidates[start:start+256]
        encoded = tokenizer([b[2] for b in batch], add_special_tokens=False,
                            truncation=False, return_attention_mask=False)['input_ids']
        for (group, prompt_hash, _), tokens in zip(batch, encoded):
            all_lengths.append(len(tokens))
            if not 32 <= len(tokens) <= 4096:
                skipped['outside_32_4096_tokens'] += 1
                continue
            split = split_for_group(group)
            pools[split].append(dict(group=group, prompt_sha256=prompt_hash,
                                    bucket=bucket(len(tokens)), input_ids=tokens))
    assert not ({r['group'] for r in pools['tune']} & {r['group'] for r in pools['eval']})
    assert not ({r['prompt_sha256'] for r in pools['tune']} & {r['prompt_sha256'] for r in pools['eval']})
    summary = dict(dataset_file=args.dataset.name, dataset_sha256=digest,
                   source_records=source_records, model=Path(args.model).name,
                   chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                   tokenizer_files_sha256={p.name: file_hash(p) for p in Path(args.model).glob('*token*') if p.is_file()},
                   skipped=dict(skipped), unique_prompt_lengths_before_filter=length_stats(all_lengths),
                   split_rule='SHA256(stage04-split-v1:original_conversation_id) modulo 5; tune=0, eval=1..4',
                   pools={split:dict(buckets=dict(Counter(r['bucket'] for r in rows)),
                                    lengths=length_stats([len(r['input_ids']) for r in rows]),
                                    unique_conversations=len({r['group'] for r in rows})) for split,rows in pools.items()},
                   note='Only intact opening user turns; exact rendered prompt deduplication; no semantic deduplication. Raw prompts and tokens stay local.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'pools.json').write_text(json.dumps(pools)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
