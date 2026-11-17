"""Build the curated SFT v3 corpus, token audit and optional judge input."""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import read_jsonl, sha256_file, technical_indicators, write_json, write_jsonl
from checks.sft_smoke_v3 import build_masked_example

DIRECTORY = ROOT / 'data/sft_v3/corpus_v1'
JUDGE_DIRECTORY = ROOT / 'judge/sft_v3_corpus_v1'
SEED = 20261117


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return ' '.join(re.findall(r'\w+', text))


def calculation_result(calculation):
    values = calculation['values']
    operations = {
        'add': lambda: sum(values),
        'subtract': lambda: values[0] - values[1],
        'multiply': lambda: math.prod(values),
        'divide': lambda: values[0] / values[1],
        'ceil_divide': lambda: math.ceil(values[0] / values[1]),
    }
    return operations[calculation['op']]()


def flatten(curated, system_prompt):
    records = []
    for group in curated['groups']:
        for case in group['cases']:
            criteria = [{'id': f'R{index}', 'text': text, 'source_ids': group['source_ids']}
                        for index, text in enumerate(group['required_criteria'], 1)]
            records.append({
                'sample_id': f'sft3_{len(records) + 1:03d}',
                'category': group['category'], 'group_id': group['group_id'],
                'split': group['split'], 'dangerous': group['dangerous'],
                'messages': [{'role': 'system', 'content': system_prompt},
                             {'role': 'user', 'content': case['question']},
                             {'role': 'assistant', 'content': case['answer']}],
                'rubric': {'required_criteria': criteria, 'critical_errors': group['critical_errors'],
                           'major_errors': group['major_errors'],
                           'acceptable_variants': ['Равнозначные формулировки и достаточный краткий ответ допустимы.'],
                           'grounding_note': group['grounding_note']},
                'author_review': {
                    'reviewer': 'Codex', 'date': '2026-11-17', 'status': 'author_reviewed',
                    'independent_review': False, 'expert_human_review': False,
                    'criteria_evidence': [{'criterion_id': criterion['id'], 'evidence': evidence}
                                          for criterion, evidence in zip(criteria, case['evidence'])],
                    'note': case['review_note'],
                },
                'calculations': case.get('calculations', []),
            })
    return records


def reference_files():
    fixed = ['data/development_v3/questions.jsonl', 'data/eval_questions.jsonl',
             'data/model_screening/questions.jsonl', 'data/sft_train.jsonl', 'data/sft_val.jsonl',
             'data/sft_v2/sft_train.jsonl', 'data/sft_v2/sft_val.jsonl']
    return [ROOT / name for name in fixed] + sorted((ROOT / 'data/raw').glob('*.jsonl'))


def check_exact_overlaps(records):
    questions = {normalize(record['messages'][1]['content']) for record in records}
    result = []
    for path in reference_files():
        existing = set()
        for row in read_jsonl(path):
            if 'question' in row:
                existing.add(normalize(row['question']))
            for message in row.get('messages', []):
                if message['role'] == 'user':
                    existing.add(normalize(message['content']))
        matches = questions & existing
        if matches:
            raise ValueError(f'Exact question overlap with {path}: {sorted(matches)}')
        result.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path), 'matches': 0})
    return result


def nearest_pairs(records, other_records):
    pairs = []
    for left in records:
        left_text = normalize(left['messages'][1]['content'])
        for right in other_records:
            similarity = SequenceMatcher(None, left_text, normalize(right['question'])).ratio()
            pairs.append({'left_id': left['sample_id'], 'right_id': right['id'],
                          'left_group': left['group_id'], 'right_group': right.get('group_id'),
                          'similarity': similarity})
    return sorted(pairs, key=lambda pair: (-pair['similarity'], pair['left_id'], pair['right_id']))[:12]


def export_judge(records, sources):
    JUDGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    system = (ROOT / 'judge/judge_prompt_v3.md').read_text(encoding='utf-8')
    schema = json.loads((ROOT / 'judge/judge_response_schema_v3.json').read_text())
    requests, mapping = [], []
    order = list(records)
    random.Random(SEED).shuffle(order)
    for record in order:
        rubric = record['rubric']
        source_ids = {source_id for criterion in rubric['required_criteria'] for source_id in criterion['source_ids']}
        payload = {
            'question': record['messages'][1]['content'], 'assistant_answer': record['messages'][2]['content'],
            'case_criteria': rubric,
            'source_notes': {key: {field: sources[key][field] for field in ['title', 'url', 'supports', 'limits']}
                             for key in sorted(source_ids)},
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        request_id = 'sft3_' + hashlib.sha256(serialized.encode()).hexdigest()[:24]
        requests.append({'request_id': request_id, 'system_prompt': system, 'user_prompt': serialized})
        mapping.append({'request_id': request_id, 'sample_id': record['sample_id'], 'split': record['split']})
    write_jsonl(JUDGE_DIRECTORY / 'judge_requests.jsonl', requests)
    write_json(JUDGE_DIRECTORY / 'judge_mapping.json', mapping)
    write_json(JUDGE_DIRECTORY / 'judge_response_schema.json', schema)
    (JUDGE_DIRECTORY / 'judge_system_prompt.md').write_text(system, encoding='utf-8')
    write_json(JUDGE_DIRECTORY / 'judge_run_config.json', {
        'status': 'prepared', 'model': 'Qwen3.8-27B', 'temperature': 0, 'max_tokens': 2048,
        'expected_requests': len(records), 'purpose': 'Independent audit of authored SFT targets',
        'system_prompt_path': 'judge_system_prompt.md', 'response_schema_path': 'judge_response_schema.json',
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'household_judge_v3', 'strict': True, 'schema': schema,
        }},
    })
    with zipfile.ZipFile(JUDGE_DIRECTORY / 'for_judge.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename in ['judge_requests.jsonl', 'judge_system_prompt.md', 'judge_response_schema.json', 'judge_run_config.json']:
            info = zipfile.ZipInfo(filename, date_time=(2026, 11, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (JUDGE_DIRECTORY / filename).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-path', help='Optional local copy of the selected model tokenizer')
    args = parser.parse_args()
    curated = json.loads((DIRECTORY / 'curated.json').read_text())
    sources = json.loads((DIRECTORY / 'sources.json').read_text())['sources']
    selection = json.loads((ROOT / 'data/sft_v3/selection.json').read_text())
    config = json.loads((ROOT / 'data/sft_v3/smoke_config.json').read_text())
    records = flatten(curated, selection['system_prompt'])
    references = check_exact_overlaps(records)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or selection['model_id'],
        **({'local_files_only': True} if args.tokenizer_path else {'revision': selection['revision']}),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_audit = []
    for record in records:
        batch, audit = build_masked_example(tokenizer, record, config['max_seq_length'])
        if any(value != -100 for value in batch['labels'][:audit['prompt_tokens']]):
            raise ValueError('Prompt was included in the loss')
        if batch['labels'][audit['unpadded_tokens'] - 1] != tokenizer.eos_token_id:
            raise ValueError('EOS was masked')
        for calculation in record['calculations']:
            if not math.isclose(calculation_result(calculation), calculation['expected'], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Arithmetic mismatch in {record['sample_id']}")
        token_audit.append(audit)
    write_jsonl(DIRECTORY / 'records.jsonl', records)
    write_jsonl(DIRECTORY / 'tokenization_audit.jsonl', token_audit)
    splits = {}
    for split in ['train', 'validation']:
        rows = [record for record in records if record['split'] == split]
        random.Random(SEED).shuffle(rows)
        write_jsonl(DIRECTORY / (split + '.jsonl'), [{'messages': row['messages']} for row in rows])
        splits[split] = {
            'n': len(rows), 'categories': dict(sorted(Counter(row['category'] for row in rows).items())),
            'dangerous': sum(row['dangerous'] for row in rows),
            'ordinary': sum(not row['dangerous'] for row in rows),
            'group_ids': sorted({row['group_id'] for row in rows}),
            'sample_ids_in_file_order': [row['sample_id'] for row in rows],
        }
    groups = [{**{key: value for key, value in group.items() if key != 'cases'},
               'sample_ids': [row['sample_id'] for row in records if row['group_id'] == group['group_id']]}
              for group in curated['groups']]
    write_json(DIRECTORY / 'groups.json', groups)
    validation_questions = [{'id': row['sample_id'], 'group_id': row['group_id'], 'question': row['messages'][1]['content']}
                            for row in records if row['split'] == 'validation']
    write_json(DIRECTORY / 'nearest_pairs.json', {
        'method': 'SequenceMatcher on normalized text, author-reviewed candidates, not a semantic duplicate detector',
        'train_validation': nearest_pairs([row for row in records if row['split'] == 'train'], validation_questions),
        'corpus_development': nearest_pairs(records, read_jsonl(ROOT / 'data/development_v3/questions.jsonl')),
    })
    review = ['# Авторская проверка 96 ответов\n\n',
              'Проверил Codex 17 ноября 2026 года. Для каждого ответа ниже сохранены '
              'критерии, фрагменты ответа и замечание к проверке. '
              'Статус author_reviewed означает авторскую проверку. Экспертная человеческая '
              'и независимая судейская оценка ещё не получены.\n\n']
    for row in records:
        review.append(f"## {row['sample_id']} · {row['category']} · {row['split']}\n\n")
        review.append(f"Группа `{row['group_id']}`, опасность `{row['dangerous']}`.\n\n")
        review.append(row['messages'][1]['content'] + '\n\n' + row['messages'][2]['content'] + '\n\n')
        for criterion, evidence in zip(row['rubric']['required_criteria'], row['author_review']['criteria_evidence']):
            review.append(f"- {criterion['id']} — {criterion['text']} Фрагмент «{evidence['evidence']}».\n")
        review.append('\n' + row['author_review']['note'] + '\n\n')
        source_ids = sorted({key for criterion in row['rubric']['required_criteria'] for key in criterion['source_ids']})
        links = [f"[{key}]({sources[key]['url']})" if sources[key]['url'] else key for key in source_ids]
        review.append('Основания приведены в реестре. ' + ', '.join(links) + '.\n\n')
    (DIRECTORY / 'review.md').write_text(''.join(review).rstrip() + '\n', encoding='utf-8')
    export_judge(records, sources)
    indicators = [technical_indicators(row['messages'][1]['content'], row['messages'][2]['content']) for row in records]
    manifest = {
        'version': curated['version'], 'created_on_moscow': curated['authored_on'], 'seed': SEED,
        'status': 'author_reviewed', 'author': 'Codex', 'expert_human_review': False,
        'independent_judge_review': 'prepared_not_run', 'n': len(records), 'groups': len(groups), 'splits': splits,
        'system_prompt_sha256': selection['system_prompt_sha256'],
        'tokenizer': {'model_id': selection['model_id'], 'revision': selection['revision'],
                      'class': type(tokenizer).__name__, 'eos_token_id': tokenizer.eos_token_id,
                      'pad_token_id': tokenizer.pad_token_id,
                      'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
        'tokenization': {'max_seq_length': config['max_seq_length'],
                         'minimum_total_tokens': min(row['unpadded_tokens'] for row in token_audit),
                         'maximum_total_tokens': max(row['unpadded_tokens'] for row in token_audit),
                         'minimum_supervised_tokens': min(row['supervised_tokens'] for row in token_audit),
                         'maximum_supervised_tokens': max(row['supervised_tokens'] for row in token_audit),
                         'truncated_examples': 0},
        'arithmetic_operations_checked': sum(len(row['calculations']) for row in records),
        'answer_words': {'minimum': min(check['word_count'] for check in indicators),
                         'median': statistics.median(check['word_count'] for check in indicators),
                         'maximum': max(check['word_count'] for check in indicators)},
        'technical_indicators': {'empty_answers': sum(check['empty'] for check in indicators),
                                 'answers_with_cjk': sum(check['cjk_characters'] > 0 for check in indicators),
                                 'answers_with_repetition': sum(check['has_repetition'] for check in indicators)},
        'external_sources': sum(bool(source['url']) for source in sources.values()),
        'source_access_counts': dict(Counter(source['verification_status'] for source in sources.values())),
        'duplicate_checks': references,
        'semantic_overlap': 'General skills overlap with development and historical data. Validation scenario families are held out from this training split. See split_review.md.',
        'data_file_sha256': {name: sha256_file(DIRECTORY / name) for name in [
            'curated.json', 'sources.json', 'records.jsonl', 'train.jsonl', 'validation.jsonl',
            'tokenization_audit.jsonl', 'groups.json', 'nearest_pairs.json', 'review.md',
            'README.md', 'split_review.md',
        ]},
        'code_sha256': {name: sha256_file(ROOT / name) for name in [
            'script/prepare_sft_v3.py', 'checks/sft_smoke_v3.py', 'data/sft_v3/corpus_v1/validate.py',
        ]},
        'judge_requests_sha256': sha256_file(JUDGE_DIRECTORY / 'judge_requests.jsonl'),
    }
    write_json(DIRECTORY / 'manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in ['n', 'groups', 'tokenization', 'arithmetic_operations_checked', 'answer_words', 'technical_indicators']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
