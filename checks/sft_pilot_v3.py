"""Paired development generation and judge export for the SFT v3 pilot."""

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from checks.screening_v3 import (
    export_artifacts, read_jsonl, technical_indicators, validate_records, write_json, write_jsonl,
)


def select_pilot_records(records, system_prompt):
    categories = {'gas', 'electricity', 'water', 'structural', 'doors_windows', 'finishes', 'appliances', 'housing'}
    if len(records) != 96 or len({row['sample_id'] for row in records}) != 96:
        raise ValueError('Expected 96 unique corpus records')
    splits = {}
    for split, count, per_category, dangerous in [('train', 80, 10, 40), ('validation', 16, 2, 8)]:
        rows = [row for row in records if row['split'] == split]
        if len(rows) != count or Counter(row['category'] for row in rows) != Counter({key: per_category for key in categories}):
            raise ValueError('Unexpected category counts in ' + split)
        if sum(row['dangerous'] for row in rows) != dangerous:
            raise ValueError('Unexpected risk balance in ' + split)
        if any(row['messages'][0]['content'] != system_prompt for row in rows):
            raise ValueError('Different system prompt in corpus')
        splits[split] = rows
    if {row['group_id'] for row in splits['train']} & {row['group_id'] for row in splits['validation']}:
        raise ValueError('A scenario group crosses train and validation')
    return splits['train'], splits['validation']


def experiment_signature(settings, selection):
    identity = {'settings': settings, 'model_id': selection['model_id'], 'revision': selection['revision'],
                'system_prompt': selection['system_prompt']}
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_pilot_records(questions, records, settings, selection, run_id, *, complete=False):
    phases = {'baseline': selection['system_prompt'], 'sft': selection['system_prompt']}
    indexed = validate_records(questions, phases, records, complete=complete)
    expected_seeds = {question['id']: settings['generation_seed'] + index
                      for index, question in enumerate(questions, 1)}
    signature = experiment_signature(settings, selection)
    for row in records:
        if row.get('run_id') != run_id or row.get('experiment_signature') != signature:
            raise ValueError('Saved answer belongs to a different experiment')
        if row.get('seed') != expected_seeds[row['id']] or row['max_new_tokens'] != settings['generation']['max_new_tokens']:
            raise ValueError('Saved generation parameters differ')
        if row.get('model_revision') != selection['revision'] or row.get('phase') != row['prompt_id']:
            raise ValueError('Saved model or phase differs')
    return indexed


def generate_pilot_phase(path, phase, questions, settings, selection, run_id, generate_one, on_saved):
    if phase not in {'baseline', 'sft'}:
        raise ValueError('Unknown generation phase')
    records = read_jsonl(path)
    indexed = validate_pilot_records(questions, records, settings, selection, run_id)
    baseline_count = sum(row['prompt_id'] == 'baseline' for row in records)
    if phase == 'sft' and baseline_count != len(questions):
        raise ValueError('Baseline must be complete before SFT generation')
    if phase == 'baseline' and baseline_count != len(questions) and any(row['prompt_id'] == 'sft' for row in records):
        raise ValueError('Cannot generate missing baseline answers after SFT answers exist')
    signature = experiment_signature(settings, selection)
    for number, question in enumerate(questions, 1):
        key = (phase, question['id'])
        if key in indexed:
            continue
        seed = settings['generation_seed'] + number
        result = generate_one(question['question'], selection['system_prompt'], seed)
        row = {
            **question, **result, 'seed': seed, 'phase': phase, 'prompt_id': phase, 'run_id': run_id,
            'model_key': selection['model_key'], 'model_revision': selection['revision'],
            'experiment_signature': signature,
            'technical': technical_indicators(question['question'], result['answer']),
        }
        candidate = records + [row]
        validate_pilot_records(questions, candidate, settings, selection, run_id)
        write_jsonl(path, candidate)
        records = candidate
        indexed[key] = row
        on_saved(row, len(records))
    return records


def export_pilot_artifacts(directory, questions, records, rubrics, sources, settings, selection,
                           run_id, judge_system, judge_schema, review_notes):
    directory = Path(directory)
    validate_pilot_records(questions, records, settings, selection, run_id, complete=True)
    judge_dir = directory / 'judge'
    judge_dir.mkdir(parents=True, exist_ok=True)
    count = export_artifacts(
        judge_dir, questions, {'baseline': selection['system_prompt'], 'sft': selection['system_prompt']},
        records, rubrics, sources, judge_system, judge_schema, run_id, settings['judge_shuffle_seed'],
    )
    summary = json.loads((judge_dir / 'automatic_summary.json').read_text(encoding='utf-8'))
    summary['quality_status'] = 'awaiting_judge_scores_and_content_review'
    write_json(directory / 'automatic_summary.json', summary)
    write_json(judge_dir / 'review_notes.json', review_notes)
    write_json(judge_dir / 'judge_run_config.json', {
        'status': 'prepared', 'model': settings['judge']['model'], 'temperature': settings['judge']['temperature'],
        'max_tokens': settings['judge']['max_tokens'], 'expected_requests': count,
        'protocol_id': 'judge_v3.0', 'system_prompt_path': 'judge_system_prompt.md',
        'response_schema_path': 'judge_response_schema.json',
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'household_judge_v3', 'strict': True, 'schema': judge_schema,
        }},
    })
    with zipfile.ZipFile(judge_dir / 'for_judge.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['judge_requests.jsonl', 'judge_system_prompt.md', 'judge_response_schema.json', 'judge_run_config.json']:
            archive.write(judge_dir / name, arcname=name)
    indexed = {(row['prompt_id'], row['id']): row for row in records}
    rubric_index = {row['id']: row for row in rubrics}
    report = ['# SFT v3 — парное сравнение development-ответов\n\n', f'Прогон `{run_id}`.\n\n',
              'Для baseline и SFT использованы одинаковые системный промпт, параметры и seed вопроса.\n\n']
    for question in questions:
        question_id = question['id']
        report.append(f"## {question_id} · {question['category']}\n\n{question['question']}\n\n")
        for criterion in rubric_index[question_id]['required_criteria']:
            report.append(f"- {criterion['id']} — {criterion['text']}\n")
        for phase, title in [('baseline', 'Исходная модель'), ('sft', 'После SFT')]:
            row = indexed[phase, question_id]
            report.append(f"\n### {title}\n\n")
            report.extend('> ' + line + '\n' for line in row['answer'].splitlines())
            if not row['answer'].strip():
                report.append('Пустой ответ.\n')
            report.append(f"\nОстановка `{row['finish_reason']}`, токенов {row['generated_tokens']}, seed {row['seed']}.\n\n")
    (directory / 'comparison.md').write_text(''.join(report).rstrip() + '\n', encoding='utf-8')
    return count
