"""Validate the fixed corpus with optional local tokenizer verification."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import read_jsonl, sha256_file, technical_indicators
from checks.sft_smoke_v3 import build_masked_example
from script.prepare_sft_v3 import calculation_result, check_exact_overlaps, flatten, normalize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(tokenizer_path=None):
    curated = json.loads((DIRECTORY / 'curated.json').read_text())
    sources = json.loads((DIRECTORY / 'sources.json').read_text())['sources']
    selection = json.loads((ROOT / 'data/sft_v3/selection.json').read_text())
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    records = read_jsonl(DIRECTORY / 'records.jsonl')
    audit = read_jsonl(DIRECTORY / 'tokenization_audit.jsonl')
    expected = flatten(curated, selection['system_prompt'])
    require(records == expected, 'Flat records differ from curated source')
    require(len(records) == manifest['n'] == 96, 'Expected 96 records')
    require(len(curated['groups']) == manifest['groups'] == 48, 'Expected 48 groups')
    require(len({group['group_id'] for group in curated['groups']}) == 48, 'Duplicate group ID')
    require(len({normalize(row['messages'][1]['content']) for row in records}) == 96, 'Duplicate question')
    require(len({normalize(row['messages'][2]['content']) for row in records}) == 96, 'Duplicate answer')
    require(manifest['system_prompt_sha256'] == selection['system_prompt_sha256'], 'System prompt hash differs')
    for filename, expected_hash in manifest['data_file_sha256'].items():
        require(sha256_file(DIRECTORY / filename) == expected_hash, f'Changed data file {filename}')
    for filename, expected_hash in manifest['code_sha256'].items():
        require(sha256_file(ROOT / filename) == expected_hash, f'Changed code file {filename}')
    require(check_exact_overlaps(records) == manifest['duplicate_checks'], 'Reference inputs changed')
    categories = {'gas', 'electricity', 'water', 'structural', 'doors_windows', 'finishes', 'appliances', 'housing'}
    index = {row['sample_id']: row for row in records}
    groups_by_split = {}
    for split, count, category_count, dangerous_count in [('train', 80, 10, 40), ('validation', 16, 2, 8)]:
        selected = [row for row in records if row['split'] == split]
        require(len(selected) == count, f'Incorrect count in {split}')
        require(Counter(row['category'] for row in selected) == Counter({key: category_count for key in categories}), 'Unbalanced categories')
        require(sum(row['dangerous'] for row in selected) == dangerous_count, 'Unbalanced risk groups')
        for category in categories:
            require(sum(row['dangerous'] for row in selected if row['category'] == category) == category_count // 2,
                    'Risk balance within a category differs')
        groups_by_split[split] = {row['group_id'] for row in selected}
        planned = manifest['splits'][split]
        require(sorted(groups_by_split[split]) == planned['group_ids'], 'Group mapping changed')
        rows_in_file = read_jsonl(DIRECTORY / (split + '.jsonl'))
        ordered = [{'messages': index[sample_id]['messages']} for sample_id in planned['sample_ids_in_file_order']]
        require(rows_in_file == ordered, 'Messages export differs from split mapping')
    require(not groups_by_split['train'] & groups_by_split['validation'], 'Scenario group crosses split')
    require(len(audit) == len(records), 'Token audit is incomplete')
    operations_checked = 0
    for row, token_row in zip(records, audit):
        sample_id = row['sample_id']
        require(type(row['dangerous']) is bool, 'Invalid danger label')
        require([message['role'] for message in row['messages']] == ['system', 'user', 'assistant'], 'Invalid role sequence')
        require(all(message['content'].strip() for message in row['messages']), 'Empty message')
        require(row['messages'][0]['content'] == selection['system_prompt'], 'Different system prompt')
        rubric, review = row['rubric'], row['author_review']
        require(2 <= len(rubric['required_criteria']) <= 3, 'Invalid criterion count')
        require(review['status'] == 'author_reviewed' and review['independent_review'] is False, 'Incorrect review attribution')
        require(bool(review['note']), 'Missing author review note')
        require(len(review['criteria_evidence']) == len(rubric['required_criteria']), 'Incomplete criterion evidence')
        for criterion, evidence in zip(rubric['required_criteria'], review['criteria_evidence']):
            require(criterion['id'] == evidence['criterion_id'], 'Criterion identity differs')
            require(evidence['evidence'] in row['messages'][2]['content'], 'Evidence is absent from answer')
            require(set(criterion['source_ids']) <= sources.keys(), 'Unknown source')
        if row['dangerous']:
            require(bool(rubric['critical_errors']), 'Missing critical error definition')
        for calculation in row['calculations']:
            require(math.isclose(calculation_result(calculation), calculation['expected'], rel_tol=1e-9, abs_tol=1e-9), 'Incorrect arithmetic')
            operations_checked += 1
        require(token_row['sample_id'] == sample_id, 'Token sample ID differs')
        require(0 < token_row['supervised_tokens'] < token_row['unpadded_tokens'] <= token_row['padded_tokens'] == 768, 'Invalid token lengths')
        require(token_row['last_supervised_token_id'] == manifest['tokenizer']['eos_token_id'], 'EOS target differs')
        indicators = technical_indicators(row['messages'][1]['content'], row['messages'][2]['content'])
        require(not indicators['empty'] and not indicators['cjk_characters'] and not indicators['has_repetition'], 'Technical content defect')
    require(operations_checked == manifest['arithmetic_operations_checked'] == 34, 'Arithmetic audit count differs')
    require(all(source['supports'] and source['limits'] for source in sources.values()), 'Source scope missing')
    judge_dir = ROOT / 'judge/sft_v3_corpus_v1'
    requests = read_jsonl(judge_dir / 'judge_requests.jsonl')
    mapping = json.loads((judge_dir / 'judge_mapping.json').read_text())
    require(len(requests) == len(mapping) == 96, 'Judge export incomplete')
    require(sha256_file(judge_dir / 'judge_requests.jsonl') == manifest['judge_requests_sha256'], 'Judge payload changed')
    mapped = {row['request_id']: row['sample_id'] for row in mapping}
    require(len(mapped) == 96 and set(mapped.values()) == set(index), 'Judge IDs are incomplete')
    for request in requests:
        payload = json.loads(request['user_prompt'])
        row = index[mapped[request['request_id']]]
        require(payload['question'] == row['messages'][1]['content'], 'Judge question differs')
        require(payload['assistant_answer'] == row['messages'][2]['content'], 'Judge target differs')
        require(payload['case_criteria'] == row['rubric'], 'Judge rubric differs')
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for row, saved_audit in zip(records, audit):
            batch, actual_audit = build_masked_example(tokenizer, row, 768)
            require(actual_audit == saved_audit, 'Recomputed token audit differs')
            require(all(label == -100 for label in batch['labels'][:actual_audit['prompt_tokens']]), 'Prompt loss mask differs')
            require(all(label == -100 for label, mask in zip(batch['labels'], batch['attention_mask']) if mask == 0), 'Padding loss mask differs')
            require(batch['labels'][actual_audit['unpadded_tokens'] - 1] == tokenizer.eos_token_id, 'EOS was masked')
    return {'status': 'OK', 'records': 96, 'train': 80, 'validation': 16,
            'scenario_groups': 48, 'cross_split_groups': 0, 'categories_in_each_split': 8,
            'dangerous': 48, 'ordinary': 48, 'arithmetic_operations_checked': operations_checked,
            'reference_files_checked': len(manifest['duplicate_checks']), 'normalized_exact_matches': 0,
            'maximum_total_tokens': max(row['unpadded_tokens'] for row in audit),
            'truncated_examples': 0, 'tokens_recomputed_now': bool(tokenizer_path),
            'author_reviewed': 96, 'expert_human_review': False, 'independent_judge_review': 'prepared_not_run'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-path', help='Recompute all masks using a local selected-model tokenizer')
    args = parser.parse_args()
    print(json.dumps(validate(args.tokenizer_path), ensure_ascii=False, indent=2))
