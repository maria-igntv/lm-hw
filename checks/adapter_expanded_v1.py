"""Checkpoint and blinded export for the fixed 0.25 LoRA development check.

This module is embedded verbatim into notebook 06, so Colab only downloads
already published, hash-pinned inputs. No model/API calls happen here.
"""

import hashlib
import json
import re
import zipfile
from pathlib import Path

from checks.screening_v3 import (
    export_artifacts, read_jsonl, sha256_file, summarize, validate_records,
    write_json, write_jsonl,
)

EXPANDED_PHASE = 'scale_0.25'
REUSED_IDS = ['dev3_01', 'dev3_07', 'dev3_15', 'dev3_21',
              'dev3_24', 'dev3_25', 'dev3_26', 'dev3_29']
PILOT_RUN = 'qwen3_4b_20261118T113449101907'
DIAGNOSTIC_RUN = 'adapter_strength_v1_20261118T200443561241'


def expanded_require(condition, message):
    if not condition:
        raise ValueError(message)


def expanded_signature(settings, selection, run_id):
    payload = {'settings': settings, 'selection': selection, 'run_id': run_id,
               'adapter_multiplier': 0.25, 'protocol': 'adapter_expanded_v1'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_expanded_history(questions, settings, selection, pilot_rows, diagnostic_rows, run_id):
    """Retain exact historical text and tokens, with explicit origin per row."""
    qids = {q['id'] for q in questions}
    expanded_require(len(qids) == len(questions) == 32, 'Expected 32 development questions')
    baseline = [r for r in pilot_rows if r['phase'] == 'baseline']
    middle = [r for r in diagnostic_rows if r['adapter_multiplier'] == 0.25]
    expanded_require(len(baseline) == 32 and {r['id'] for r in baseline} == qids,
                     'Incomplete historical baseline')
    expanded_require(len(middle) == 8 and {r['question_id'] for r in middle} == set(REUSED_IDS),
                     'Incomplete historical scale 0.25')
    result = []
    signature = expanded_signature(settings, selection, run_id)
    for phase, source_run, rows in [('baseline', PILOT_RUN, baseline),
                                     (EXPANDED_PHASE, DIAGNOSTIC_RUN, middle)]:
        for row in rows:
            qid = row.get('question_id', row.get('id'))
            result.append({**row, 'id': qid, 'phase': phase, 'prompt_id': phase,
                           'run_id': run_id, 'source_run_id': source_run, 'origin': 'reused',
                           'model_revision': selection['revision'],
                           'adapter_multiplier': 0.0 if phase == 'baseline' else 0.25,
                           'eos_token_ids': [151645, 151643],
                           'experiment_signature': signature})
    validate_expanded_records(questions, settings, selection, run_id, result, result)
    return result


def validate_expanded_records(questions, settings, selection, run_id, rows, history, *, complete=False):
    phases = {'baseline': selection['system_prompt'], EXPANDED_PHASE: selection['system_prompt']}
    indexed = validate_records(questions, phases, rows, complete=complete)
    expected_history = {(r['prompt_id'], r['id']): r for r in history}
    expanded_require(len(expected_history) == len(history) == 40, 'Expected 40 historical answers')
    signature = expanded_signature(settings, selection, run_id)
    seeds = {q['id']: settings['generation_seed'] + i for i, q in enumerate(questions, 1)}
    for key, expected in expected_history.items():
        expanded_require(indexed.get(key) == expected, 'Historical answer was changed or removed: ' + str(key))
    for key, row in indexed.items():
        expanded_require(row.get('experiment_signature') == signature and row.get('run_id') == run_id,
                         'Saved answer belongs to a different experiment')
        expanded_require(row.get('seed') == seeds[row['id']]
                         and row['max_new_tokens'] == settings['generation']['max_new_tokens'],
                         'Saved seed or generation limit differs')
        expanded_require(row.get('model_revision') == selection['revision']
                         and row.get('phase') == row['prompt_id'], 'Saved model or phase differs')
        expected_scale = 0.0 if row['prompt_id'] == 'baseline' else 0.25
        expanded_require(row.get('adapter_multiplier') == expected_scale, 'Saved scale differs')
        if key not in expected_history:
            expanded_require(row['prompt_id'] == EXPANDED_PHASE and row.get('origin') == 'new'
                             and row.get('source_run_id') == run_id, 'Unexpected new answer provenance')
    return indexed


def generate_expanded_remaining(path, questions, settings, selection, run_id, history, generate_one, on_saved):
    """Resume without regenerating or choosing among already saved answers."""
    path = Path(path)
    rows = read_jsonl(path) if path.exists() else list(history)
    indexed = validate_expanded_records(questions, settings, selection, run_id, rows, history)
    write_jsonl(path, rows)
    signature = expanded_signature(settings, selection, run_id)
    for number, question in enumerate(questions, 1):
        key = (EXPANDED_PHASE, question['id'])
        if key in indexed:
            continue
        seed = settings['generation_seed'] + number
        answer = generate_one(question, selection['system_prompt'], seed)
        row = {**question, **answer, 'seed': seed, 'phase': EXPANDED_PHASE,
               'prompt_id': EXPANDED_PHASE, 'adapter_multiplier': 0.25,
               'run_id': run_id, 'source_run_id': run_id, 'origin': 'new',
               'model_revision': selection['revision'], 'experiment_signature': signature}
        candidate = rows + [row]
        validate_expanded_records(questions, settings, selection, run_id, candidate, history)
        write_jsonl(path, candidate)
        rows = candidate
        indexed[key] = row
        on_saved(row, sum(r['origin'] == 'new' for r in rows))
    validate_expanded_records(questions, settings, selection, run_id, rows, history, complete=True)
    return rows


def expanded_repeated_span(text):
    words = re.findall(r'\w+', text.casefold())
    for width in range(3, min(128, len(words) // 3) + 1):
        for start in range(len(words) - width * 3 + 1):
            block = words[start:start + width]
            if words[start + width:start + 3 * width] == block * 2:
                return {'period_words': width, 'start_word': start}
    return None


def export_expanded_reuse(directory, source_root, history, rubrics, sources):
    """Link 40 unchanged answers per judge to the exact original judge payload."""
    root = Path(source_root)
    rubric_index = {r['id']: r for r in rubrics}
    result = []
    for judge, model_name in [('qwen', 'Qwen3.8-27B'), ('gemma', 'gemma-4-31B-it')]:
        old = read_jsonl(root / f'results/sft_v3_pilot/judges/{judge}_scores.jsonl')
        middle = read_jsonl(root / f'results/adapter_strength_v1/judges/{judge}_all_scales.jsonl')
        prior_requests = {r['request_id']: r for r in read_jsonl(root / 'judge/sft_v3_pilot/judge_requests.jsonl')}
        diagnostic_requests = {r['request_id']: r for r in read_jsonl(root / 'judge/adapter_strength_v1/judge_requests.jsonl')}
        base = {r['question_id']: r for r in old if r['phase'] == 'baseline'}
        scaled = {r['question_id']: r for r in middle if r['adapter_multiplier'] == 0.25}
        expanded_require(len(base) == 32 and len(scaled) == 8, 'Prior score coverage differs')
        for answer in history:
            is_base = answer['prompt_id'] == 'baseline'
            score = (base if is_base else scaled)[answer['id']]
            request = (prior_requests if is_base else diagnostic_requests)[score['request_id']]
            rubric = rubric_index[answer['id']]
            basis_ids = sorted({key for c in rubric['required_criteria'] for key in c['source_ids']})
            basis = {key: {f: sources[key][f] for f in ('title', 'url', 'supports', 'limits')} for key in basis_ids}
            expected = {'question': answer['question'], 'assistant_answer': answer['answer'],
                        'case_criteria': rubric, 'source_notes': basis}
            expanded_require(json.loads(request['user_prompt']) == expected, 'Prior judged payload differs')
            expanded_require(score['judge_model'] == model_name and json.loads(score['raw_response']) == score['score'],
                             'Prior judge model or raw score differs')
            result.append({'judge': judge, 'judge_model': model_name,
                           'question_id': answer['id'], 'prompt_id': answer['prompt_id'],
                           'prior_request_id': score['request_id'], 'source_candidate_run': answer['source_run_id'],
                           'source_judge_run': score.get('source_run_id', score.get('service_batch')),
                           'score': score['score'], 'raw_response': score['raw_response'],
                           'semantic_state': score['semantic_state'],
                           'source_scores_file': f'results/sft_v3_pilot/judges/{judge}_scores.jsonl' if is_base
                           else f'results/adapter_strength_v1/judges/{judge}_all_scales.jsonl',
                           'judged_payload_verified_identical': True})
    expanded_require(len(result) == 80, 'Expected 40 reused scores per judge')
    write_jsonl(Path(directory) / 'reused_judge_scores.jsonl', result)
    return result


def export_expanded_artifacts(directory, questions, rows, history, settings, selection, run_id,
                              rubrics, sources, judge_system, judge_schema):
    directory = Path(directory)
    validate_expanded_records(questions, settings, selection, run_id, rows, history, complete=True)
    new_rows = [r for r in rows if r['origin'] == 'new']
    new_ids = {r['id'] for r in new_rows}
    expanded_require(len(new_rows) == len(new_ids) == 24, 'Expected 24 new answers')
    new_questions = [q for q in questions if q['id'] in new_ids]
    new_rubrics = [r for r in rubrics if r['id'] in new_ids]
    judge_dir = directory / 'judge'
    judge_dir.mkdir(exist_ok=True)
    count = export_artifacts(judge_dir, new_questions, {EXPANDED_PHASE: selection['system_prompt']},
                             new_rows, new_rubrics, sources, judge_system, judge_schema,
                             run_id, settings['judge_shuffle_seed'])
    config = {'status': 'prepared_not_scored', 'protocol_id': 'judge_v3.0',
              'model': 'Qwen3.8-27B', 'optional_second_model': 'gemma-4-31B-it',
              'temperature': 0, 'max_tokens': 2048, 'expected_requests': count,
              'system_prompt_path': 'judge_system_prompt.md',
              'response_schema_path': 'judge_response_schema.json',
              'response_format': {'type': 'json_schema', 'json_schema': {
                  'name': 'household_judge_v3', 'strict': True, 'schema': judge_schema}},
              'retry_policy': 'Only retry invalid/schema-inconsistent responses; preserve all attempts, errors and actual run_id. Never retry a valid low score.',
              'required_return': '24 scores with request_id, score, raw_response, judge_model, run_id; all raw attempts, error journal and model/parameter/time metadata.'}
    write_json(judge_dir / 'judge_run_config.json', config)
    with zipfile.ZipFile(judge_dir / 'for_judge.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['judge_requests.jsonl', 'judge_system_prompt.md',
                     'judge_response_schema.json', 'judge_run_config.json']:
            archive.write(judge_dir / name, arcname=name)
    summary = summarize(questions, {'baseline': selection['system_prompt'], EXPANDED_PHASE: selection['system_prompt']}, rows)
    for phase, group in summary['by_prompt'].items():
        phase_rows = [r for r in rows if r['prompt_id'] == phase]
        group['repeat_3_to_128_words_ids'] = [r['id'] for r in phase_rows if expanded_repeated_span(r['answer'])]
        group['repeat_after_list_counter_removal_ids'] = [r['id'] for r in phase_rows
            if expanded_repeated_span(re.sub(r'(?m)^\s*\d+[.)]\s+', '', r['answer']))]
    summary.update({'new_answers': 24, 'reused_answers': 40,
                    'quality_status': 'awaiting_external_judge_and_content_review',
                    'scope': 'Expanded development; not an independent final test.'})
    write_json(directory / 'automatic_summary.json', summary)
    indexed = {(r['prompt_id'], r['id']): r for r in rows}
    rubric_index = {r['id']: r for r in rubrics}
    report = ['# Полный development: исходная модель и LoRA 0,25\n\n',
              '32 пары, 24 новых ответа, 40 переиспользованных. Это development, не финальный тест.\n\n']
    for question in questions:
        qid = question['id']
        report.append(f"## {qid} · {question['category']}\n\n{question['question']}\n\n")
        report.extend(f"- {c['id']}: {c['text']}\n" for c in rubric_index[qid]['required_criteria'])
        for phase in ['baseline', EXPANDED_PHASE]:
            row = indexed[phase, qid]
            report.append(f"\n### {phase}\n\n")
            report.extend('> ' + line + '\n' for line in row['answer'].splitlines())
            report.append(f"\nТокенов: {row['generated_tokens']}; {row['finish_reason']}; origin={row['origin']}.\n\n")
    (directory / 'comparison.md').write_text(''.join(report), encoding='utf-8')
    return count
