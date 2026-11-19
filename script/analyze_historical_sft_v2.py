"""Recompute the published historical v2 metrics without GPU or model calls."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.evaluation import compare_answers
from checks.screening_v3 import read_jsonl, sha256_file


def main():
    directory = ROOT / 'results/sft_v2'
    signature = json.loads((directory / 'evaluation_signature.json').read_text())
    for name, key in [('data/eval_questions.jsonl', 'questions_sha256'),
                      ('results/baseline_answers.jsonl', 'baseline_sha256'),
                      ('checks/autocheck.py', 'checker_sha256')]:
        if sha256_file(ROOT / name) != signature[key]:
            raise ValueError('Historical comparison input differs: ' + name)
    summaries, transitions, _ = compare_answers(read_jsonl(ROOT / 'data/eval_questions.jsonl'),
        read_jsonl(ROOT / 'results/baseline_answers.jsonl'), read_jsonl(directory / 'sft_answers.jsonl'))
    actual = {'summaries': summaries, 'transitions': transitions}
    if actual != json.loads((directory / 'metrics_comparison.json').read_text()):
        raise ValueError('Recomputed v2 metrics differ')
    print(json.dumps({'verification': 'PASS', **actual}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
