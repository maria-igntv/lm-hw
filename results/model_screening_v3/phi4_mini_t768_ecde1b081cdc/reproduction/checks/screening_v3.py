"""Persistence, technical indicators and judge export for the v3 screening."""

import hashlib
import json
import random
import re
import statistics
import unicodedata
from pathlib import Path

from checks.autocheck import detect_danger_topics, has_repetition, has_warning
from checks.evaluation import index_records


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, records):
    """Replace a whole small checkpoint so interrupted writes keep the previous one."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records),
        encoding='utf-8',
    )
    temporary.replace(path)


def finish_reason(token_ids, eos_token_ids, max_new_tokens):
    if token_ids and token_ids[-1] in eos_token_ids:
        return 'eos'
    if len(token_ids) >= max_new_tokens:
        return 'length'
    return 'other'


def technical_indicators(question, answer):
    letters = [character for character in answer if character.isalpha()]
    cyrillic = sum('CYRILLIC' in unicodedata.name(character, '') for character in letters)
    cjk = re.findall('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]', answer)
    return {
        'empty': not answer.strip(),
        'word_count': len(re.findall(r'\w+', answer)),
        'cyrillic_letter_ratio': cyrillic / len(letters) if letters else None,
        'cjk_characters': len(cjk),
        'has_repetition': has_repetition(answer),
        'warning_marker_found': has_warning(answer),
        'regex_danger_topics': detect_danger_topics(question),
    }


def validate_records(questions, prompt_ids, records, *, complete=False):
    """The key is (prompt_id, question_id), including empty generated answers."""
    expected = index_records(questions)
    expected_keys = {(prompt_id, question_id) for prompt_id in prompt_ids for question_id in expected}
    actual = {}
    for record in records:
        key = (record['prompt_id'], record['id'])
        if key not in expected_keys:
            raise ValueError(f'Unexpected answer key: {key}')
        if key in actual:
            raise ValueError(f'Duplicate answer key: {key}')
        question = expected[record['id']]
        for field in ('question', 'category', 'dangerous'):
            if record.get(field) != question[field]:
                raise ValueError(f'{key}: different {field}')
        if not isinstance(record.get('answer'), str):
            raise ValueError(f'{key}: invalid answer')
        token_ids = record['generated_token_ids']
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError(f'{key}: invalid token IDs')
        if record['generated_tokens'] != len(token_ids):
            raise ValueError(f'{key}: token count differs')
        if record['finish_reason'] != finish_reason(token_ids, record['eos_token_ids'], record['max_new_tokens']):
            raise ValueError(f'{key}: incorrect finish reason')
        actual[key] = record
    if complete and set(actual) != expected_keys:
        raise ValueError(f'Expected {len(expected_keys)} answers, found {len(actual)}')
    return actual


def summarize(questions, prompts, records):
    actual = validate_records(questions, prompts, records)
    summaries = {}
    for prompt_id in prompts:
        group = [record for record in records if record['prompt_id'] == prompt_id]
        checks = [technical_indicators(record['question'], record['answer']) for record in group]
        dangerous = [(record, check) for record, check in zip(group, checks) if record['dangerous']]
        ordinary = [(record, check) for record, check in zip(group, checks) if not record['dangerous']]
        summaries[prompt_id] = {
            'n': len(group),
            'expected_n': len(questions),
            'empty_answers': sum(check['empty'] for check in checks),
            'length_stops': sum(record['finish_reason'] == 'length' for record in group),
            'eos_stops': sum(record['finish_reason'] == 'eos' for record in group),
            'other_stops': sum(record['finish_reason'] == 'other' for record in group),
            'answers_with_cjk': sum(check['cjk_characters'] > 0 for check in checks),
            'answers_with_repetition': sum(check['has_repetition'] for check in checks),
            'dangerous_n': len(dangerous),
            'dangerous_with_warning_marker': sum(check['warning_marker_found'] for _, check in dangerous),
            'ordinary_n': len(ordinary),
            'ordinary_with_warning_marker': sum(check['warning_marker_found'] for _, check in ordinary),
            'mean_generated_tokens': statistics.mean(record['generated_tokens'] for record in group) if group else None,
            'median_word_count': statistics.median(check['word_count'] for check in checks) if group else None,
            'total_generation_seconds': sum(record['elapsed_seconds'] for record in group),
            'max_peak_allocated_gib': max((record['peak_allocated_gib'] for record in group), default=None),
        }
    return {
        'n': len(actual), 'expected_n': len(questions) * len(prompts),
        'complete': len(actual) == len(questions) * len(prompts),
        'by_prompt': summaries,
        'interpretation': 'Технические индикаторы и regex-маркеры. Содержательная оценка выполняется отдельно по rubric.jsonl.',
    }


def export_artifacts(out_dir, questions, prompts, records, rubrics, sources, judge_system, judge_schema, run_id, seed):
    out_dir = Path(out_dir)
    validate_records(questions, prompts, records)
    rubric_index = index_records(rubrics)
    question_ids = {question['id'] for question in questions}
    if set(rubric_index) != question_ids:
        raise ValueError('Rubric IDs differ from development questions')
    if any(not rubric['required_criteria'] for rubric in rubrics):
        raise ValueError('Missing criteria')
    for rubric in rubrics:
        for criterion in rubric['required_criteria']:
            if not set(criterion['source_ids']) <= sources.keys():
                raise ValueError(f"Unknown basis for {rubric['id']}")

    requests = []
    mapping = []
    ordered = sorted(records, key=lambda record: (record['id'], record['prompt_id']))
    random.Random(seed).shuffle(ordered)
    for record in ordered:
        rubric = rubric_index[record['id']]
        basis_ids = sorted({key for criterion in rubric['required_criteria'] for key in criterion['source_ids']})
        basis = {
            key: {field: sources[key][field] for field in ('title', 'url', 'supports', 'limits')}
            for key in basis_ids
        }
        payload = {
            'question': record['question'],
            'assistant_answer': record['answer'],
            'case_criteria': rubric,
            'source_notes': basis,
        }
        identity = json.dumps([run_id, record['prompt_id'], record['id']], ensure_ascii=False)
        request_id = 'v3_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
        requests.append({
            'request_id': request_id,
            'system_prompt': judge_system,
            'user_prompt': json.dumps(payload, ensure_ascii=False),
        })
        mapping.append({
            'request_id': request_id, 'question_id': record['id'],
            'prompt_id': record['prompt_id'], 'run_id': run_id,
        })

    write_json(out_dir / 'automatic_summary.json', summarize(questions, prompts, records))
    write_jsonl(out_dir / 'judge_requests.jsonl', requests)
    write_json(out_dir / 'judge_mapping.json', mapping)
    write_json(out_dir / 'judge_response_schema.json', judge_schema)
    (out_dir / 'judge_system_prompt.md').write_text(judge_system, encoding='utf-8')
    report = ['# Ответы development v3\n\n', f'Прогон `{run_id}`. Сохранено {len(records)} ответов.\n\n']
    for prompt_id in prompts:
        report.append(f'## Промпт {prompt_id}\n\n')
        for record in sorted(records, key=lambda item: item['id']):
            if record['prompt_id'] != prompt_id:
                continue
            report.append(f"### {record['id']}\n\n{record['question']}\n\n")
            report.extend('> ' + line + '\n' for line in record['answer'].splitlines())
            if not record['answer'].strip():
                report.append('Пустой ответ.\n')
            report.append(f"\nОстановка `{record['finish_reason']}`, токенов {record['generated_tokens']}.\n\n")
    (out_dir / 'answers.md').write_text(''.join(report), encoding='utf-8')
    return len(requests)
