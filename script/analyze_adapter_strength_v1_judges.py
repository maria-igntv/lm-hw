"""Validate middle-scale judge exports, reuse endpoint scores and recompute paired metrics."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import read_jsonl, sha256_file, write_json, write_jsonl
from script.analyze_judges_v3 import condition_summary, semantic_state

RUN = 'adapter_strength_v1_20261118T200443561241'
PRIOR_RUN = 'qwen3_4b_20261118T113449101907'
JUDGE_DIR = ROOT / 'judge/adapter_strength_v1'
PRIOR_JUDGE_DIR = ROOT / 'judge/sft_v3_pilot'
PRIOR_SCORE_DIR = ROOT / 'results/sft_v3_pilot/judges'
OUTPUT = ROOT / 'results/adapter_strength_v1/judges'
SCALES = [0.0, 0.25, 0.5, 1.0]
JUDGES = {'qwen': 'Qwen3.8-27B', 'gemma': 'gemma-4-31B-it'}
COMPARISONS = [('0.25_vs_0', 0.25, 0.0), ('0.5_vs_0', 0.5, 0.0),
               ('0.25_vs_1', 0.25, 1.0), ('0.5_vs_1', 0.5, 1.0)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def scale_from_prompt(prompt_id):
    require(prompt_id.startswith('scale_'), 'Unknown middle-scale prompt ID')
    return float(prompt_id.removeprefix('scale_'))


def validate_score(score, request, validator):
    validator.validate(score)
    payload = json.loads(request['user_prompt'])
    expected = Counter(item['id'] for item in payload['case_criteria']['required_criteria'])
    actual = Counter(item['criterion_id'] for item in score['criteria_checks'])
    require(actual == expected, f"{request['request_id']}: required criterion coverage differs")
    require(not score['critical_errors'] or score['safety'] == 1,
            f"{request['request_id']}: critical_errors requires safety=1")


def build_context():
    requests_list = read_jsonl(JUDGE_DIR / 'judge_requests.jsonl')
    requests = {row['request_id']: row for row in requests_list}
    require(len(requests) == len(requests_list) == 16, 'Expected 16 unique new judge requests')
    mapping_list = load_json(JUDGE_DIR / 'judge_mapping.json')
    mapping = {row['request_id']: row for row in mapping_list}
    require(set(mapping) == set(requests) and len(mapping_list) == 16, 'New request mapping differs')
    prior_requests_list = read_jsonl(PRIOR_JUDGE_DIR / 'judge_requests.jsonl')
    prior_requests = {row['request_id']: row for row in prior_requests_list}
    prior_mapping_list = load_json(PRIOR_JUDGE_DIR / 'judge_mapping.json')
    prior_mapping = {row['request_id']: row for row in prior_mapping_list}
    require(len(prior_requests) == len(prior_mapping) == 64, 'Prior judge package is incomplete')
    answers_list = read_jsonl(ROOT / 'results/adapter_strength_v1' / RUN / 'answers.jsonl')
    answers = {(row['adapter_multiplier'], row['question_id']): row for row in answers_list}
    require(len(answers) == len(answers_list) == 32, 'Expected 32 unique adapter-strength answers')
    prior_answers_list = read_jsonl(ROOT / 'results/sft_v3_pilot' / PRIOR_RUN / 'answers.jsonl')
    prior_answers = {(row['phase'], row['id']): row for row in prior_answers_list}
    require(len(prior_answers) == len(prior_answers_list) == 64, 'Prior answer set is incomplete')
    questions = {row['id']: row for row in read_jsonl(ROOT / 'data/development_v3/questions.jsonl')}
    schema = load_json(JUDGE_DIR / 'judge_response_schema.json')
    validator = Draft202012Validator(schema)
    return {
        'requests': requests, 'mapping': mapping, 'prior_requests': prior_requests,
        'prior_mapping': prior_mapping, 'answers': answers, 'prior_answers': prior_answers,
        'questions': questions, 'schema': schema, 'validator': validator,
    }


def verify_frozen_inputs(context):
    system_prompt = (JUDGE_DIR / 'judge_system_prompt.md').read_text(encoding='utf-8')
    prior_system_prompt = (PRIOR_JUDGE_DIR / 'judge_system_prompt.md').read_text(encoding='utf-8')
    require(system_prompt == prior_system_prompt, 'Judge system prompts differ')
    require(context['schema'] == load_json(PRIOR_JUDGE_DIR / 'judge_response_schema.json'),
            'Judge response schemas differ')
    for request_id, request in context['requests'].items():
        require(request['system_prompt'] == system_prompt, f'{request_id}: system prompt differs')
        link = context['mapping'][request_id]
        scale = scale_from_prompt(link['prompt_id'])
        require(scale in [0.25, 0.5] and link['run_id'] == RUN, f'{request_id}: mapping differs')
        answer = context['answers'][(scale, link['question_id'])]
        payload = json.loads(request['user_prompt'])
        require(payload['question'] == answer['question'], f'{request_id}: question differs')
        require(payload['assistant_answer'] == answer['answer'], f'{request_id}: answer differs')
        question = context['questions'][link['question_id']]
        require(answer['question'] == question['question'], f'{request_id}: development question differs')
        require(payload['case_criteria']['id'] == link['question_id'], f'{request_id}: criterion case differs')
    for request_id, request in context['prior_requests'].items():
        require(request['system_prompt'] == prior_system_prompt, f'{request_id}: prior system prompt differs')
        link = context['prior_mapping'][request_id]
        answer = context['prior_answers'][(link['prompt_id'], link['question_id'])]
        payload = json.loads(request['user_prompt'])
        require(payload['question'] == answer['question'] and payload['assistant_answer'] == answer['answer'],
                f'{request_id}: prior judge payload differs from candidate')


def public_run(run):
    fields = ['service_batch', 'key', 'id', 'model', 'status', 'count', 'output_count', 'failed_count',
              'created_at', 'updated_at', 'parameters', 'response_format', 'catalog', 'usage',
              'worker_build_version', 'system_prompt_sha256', 'per_response_metadata',
              'initial_valid_count', 'final_valid_count', 'total_api_calls_including_retries',
              'replacement_request_ids']
    result = {field: run[field] for field in fields if field in run}
    if 'retry_runs' in run:
        result['retry_runs'] = [public_run(child) for child in run['retry_runs']]
    return result


def public_input_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return 'external_input/' + path.name


def verify_endpoints(context):
    verify_frozen_inputs(context)
    reuse = load_json(JUDGE_DIR / 'endpoint_score_reuse.json')
    require(len(reuse) == 16, 'Expected 16 endpoint reuse links')
    require({(row['multiplier'], row['question_id']) for row in reuse}
            == {(scale, qid) for scale in [0.0, 1.0] for qid in sorted({qid for _, qid in context['answers']})},
            'Endpoint scale/question coverage differs')
    prior_runs = load_json(PRIOR_SCORE_DIR / 'runs.json')
    require(len(prior_runs) == 2, 'Expected two prior judge runs')
    run_by_model = {run['model']: run for run in prior_runs}
    records = []
    normalized = {key: {} for key in JUDGES}
    for link in reuse:
        scale, question_id = link['multiplier'], link['question_id']
        phase = 'baseline' if scale == 0.0 else 'sft'
        require(link['generated_token_ids_identical'] is True and link['prior_candidate_run'] == PRIOR_RUN,
                'Endpoint provenance declaration differs')
        prior_answer = context['prior_answers'][(phase, question_id)]
        answer = context['answers'][(scale, question_id)]
        require(answer['generated_token_ids'] == prior_answer['generated_token_ids'],
                f'{scale}/{question_id}: generated tokens differ')
        require(answer['answer'] == prior_answer['answer'] and answer['question'] == prior_answer['question'],
                f'{scale}/{question_id}: endpoint text differs')
        request_id = link['prior_request_id']
        require(request_id in context['prior_requests'] and request_id in context['prior_mapping'],
                f'{scale}/{question_id}: unknown prior request')
        old_link = context['prior_mapping'][request_id]
        require(old_link == {'request_id': request_id, 'question_id': question_id,
                             'prompt_id': phase, 'run_id': PRIOR_RUN}, 'Prior request mapping differs')
        old_payload = json.loads(context['prior_requests'][request_id]['user_prompt'])
        require(old_payload['question'] == answer['question']
                and old_payload['assistant_answer'] == answer['answer'],
                f'{scale}/{question_id}: prior judged text differs')
        for new_id, new_request in context['requests'].items():
            if context['mapping'][new_id]['question_id'] == question_id:
                new_payload = json.loads(new_request['user_prompt'])
                require(new_payload['case_criteria'] == old_payload['case_criteria']
                        and new_payload['source_notes'] == old_payload['source_notes'],
                        f'{question_id}: criteria or sources differ across scales')
        for judge, expected_model in JUDGES.items():
            score_path = ROOT / link['judges'][judge]['scores_path']
            require(score_path == PRIOR_SCORE_DIR / f'{judge}_scores.jsonl', 'Unexpected endpoint score source')
        records.append({'multiplier': scale, 'question_id': question_id, 'phase': phase,
                        'prior_request_id': request_id, 'generated_tokens_identical': True,
                        'judged_text_identical': True, 'criteria_and_sources_identical': True})
    for judge, expected_model in JUDGES.items():
        rows = read_jsonl(PRIOR_SCORE_DIR / f'{judge}_scores.jsonl')
        index = {row['request_id']: row for row in rows}
        require(len(index) == len(rows) == 64, f'{judge}: prior scores are incomplete')
        run = run_by_model[expected_model]
        require(run['model'] == expected_model and run['status'] == 'success'
                and run['parameters']['temperature'] == 0 and run['parameters']['max_tokens'] == 2048,
                f'{judge}: prior judge protocol differs')
        require(run['response_format']['json_schema']['schema'] == context['schema'],
                f'{judge}: prior judge schema differs')
        for link in reuse:
            request_id = link['prior_request_id']
            row = index[request_id]
            require(row['judge_model'] == expected_model and row['service_batch'] == run['service_batch'],
                    f'{judge}/{request_id}: model or service batch differs')
            require(row['candidate'] == context['prior_mapping'][request_id],
                    f'{judge}/{request_id}: candidate mapping differs')
            validate_score(row['score'], context['prior_requests'][request_id], context['validator'])
            require(json.loads(row['raw_response']) == row['score'],
                    f'{judge}/{request_id}: prior parsed/raw score differs')
            require(semantic_state(row['score']) == row['semantic_state']
                    == link['judges'][judge]['semantic_state'],
                    f'{judge}/{request_id}: prior semantic state differs')
            normalized[judge][request_id] = row
    return records, normalized, prior_runs


def find_new_runs(input_dir):
    path = input_dir / 'runs.json'
    require(path.exists(), f'Missing {path}; new judge runs have not been supplied')
    value = load_json(path)
    runs = value['runs'] if isinstance(value, dict) and 'runs' in value else value
    require(isinstance(runs, list) and len(runs) == 2, 'Expected exactly two new judge run records')
    result = {}
    for judge, model in JUDGES.items():
        selected = [run for run in runs if run.get('model') == model]
        require(len(selected) == 1, f'{judge}: expected one run for exact model {model}')
        run = selected[0]
        require(run.get('status') == 'success', f'{judge}: new run is not successful')
        require(run['parameters']['temperature'] == 0 and run['parameters']['max_tokens'] == 2048,
                f'{judge}: new inference parameters differ')
        result[judge] = run
    return result


def validate_new_scores(context, input_dir, runs):
    normalized = {}
    raw_normalized = {}
    for judge, expected_model in JUDGES.items():
        scores_path = input_dir / f'{judge}_results.jsonl'
        raw_path = input_dir / 'raw' / f'{judge}_new_output.jsonl'
        require(scores_path.exists() and raw_path.exists(),
                f'{judge}: missing new scores or raw output in {input_dir}')
        scores = read_jsonl(scores_path)
        raw_rows = read_jsonl(raw_path)
        score_index = {row['request_id']: row for row in scores}
        raw_index = {row['uid']: row for row in raw_rows}
        require(len(score_index) == len(scores) == 16, f'{judge}: missing or duplicate new scores')
        require(len(raw_index) == len(raw_rows) == 16, f'{judge}: missing or duplicate raw responses')
        require(set(score_index) == set(raw_index) == set(context['requests']),
                f'{judge}: new request coverage differs')
        run = runs[judge]
        require(run['response_format']['json_schema']['schema'] == context['schema'],
                f'{judge}: effective new schema differs')
        expected_prompt_hash = sha256_text((JUDGE_DIR / 'judge_system_prompt.md').read_text(encoding='utf-8'))
        require(run.get('system_prompt_sha256') == expected_prompt_hash,
                f'{judge}: effective system prompt hash missing or differs')
        run_index = {item['id']: item for item in [run, *run.get('retry_runs', [])]}
        require(len(run_index) == 1 + len(run.get('retry_runs', [])), f'{judge}: duplicate run IDs')
        for item in run_index.values():
            require(item['model'] == expected_model and item['status'] == 'success',
                    f'{judge}: retry model or status differs')
            require(item['parameters'] == run['parameters'] and item['response_format'] == run['response_format']
                    and item.get('system_prompt_sha256') == expected_prompt_hash and item.get('catalog') == run.get('catalog'),
                    f'{judge}: retry prompt, schema or parameters differ')
        result = []
        for request_id, row in score_index.items():
            require(row.get('judge_model') == expected_model, f'{judge}/{request_id}: judge model differs')
            require(row.get('run_id') in run_index, f'{judge}/{request_id}: unknown actual source run')
            if 'candidate' in row:
                require(row['candidate'] == context['mapping'][request_id],
                        f'{judge}/{request_id}: candidate mapping differs')
            score = row.get('score')
            require(isinstance(score, dict), f'{judge}/{request_id}: missing parsed score')
            validate_score(score, context['requests'][request_id], context['validator'])
            raw_response = raw_index[request_id].get('model_response')
            require(isinstance(raw_response, str), f'{judge}/{request_id}: missing raw model response')
            require(row.get('raw_response') == raw_response and json.loads(raw_response) == score,
                    f'{judge}/{request_id}: parsed and raw responses differ')
            link = context['mapping'][request_id]
            scale = scale_from_prompt(link['prompt_id'])
            answer = context['answers'][(scale, link['question_id'])]
            question = context['questions'][link['question_id']]
            flags = []
            if score['unsupported_specifics'] and not score['critical_errors'] and not score['major_errors']:
                flags.append('unsupported_specifics_without_recorded_error')
            if answer['finish_reason'] == 'length' and semantic_state(score) == 'pass':
                flags.append('judge_accepts_truncated_candidate')
            result.append({
                'request_id': request_id, 'judge_model': expected_model,
                'question_id': link['question_id'], 'adapter_multiplier': scale,
                'category': question['category'], 'dangerous': question['dangerous'],
                'candidate_run_id': RUN, 'candidate_finish_reason': answer['finish_reason'],
                'score': score, 'semantic_state': semantic_state(score),
                'semantic_pass': semantic_state(score) == 'pass', 'review_flags': flags,
                'raw_response': raw_response, 'score_origin': 'new',
                'source_file': public_input_path(scores_path),
                'source_run_id': row['run_id'],
            })
        normalized[judge] = sorted(result, key=lambda row: (row['adapter_multiplier'], row['question_id']))
        raw_normalized[judge] = sorted(raw_rows, key=lambda row: row['uid'])
    return normalized, raw_normalized


def verify_attempt_history(context, input_dir, runs):
    """Verify any supplied retry transcript without replacing valid low scores."""
    result = {'api_calls': 0, 'invalid_attempts': 0, 'by_judge': {}}
    for judge, run in runs.items():
        final = {row['uid']: row for row in read_jsonl(input_dir / 'raw' / f'{judge}_new_output.jsonl')}
        retries = run.get('retry_runs', [])
        initial_path = input_dir / 'raw' / f'{judge}_initial_output.jsonl'
        if not initial_path.exists():
            require(not retries, f'{judge}: retry metadata has no initial transcript')
            initial_path = input_dir / 'raw' / f'{judge}_new_output.jsonl'
        initial = read_jsonl(initial_path)
        require(len(initial) == len(final) == 16 and {r['uid'] for r in initial} == set(final),
                f'{judge}: initial transcript coverage differs')
        retry_files = sorted((input_dir / 'raw').glob(f'{judge}_retry*_output.jsonl'))
        require(len(retry_files) == len(retries), f'{judge}: retry transcript count differs')
        batches = [(run, initial)] + [(child, read_jsonl(path)) for child, path in zip(retries, retry_files)]
        accepted, invalid, calls = {}, [], 0
        for batch, rows in batches:
            require(len(rows) == batch['count'] and len({r['uid'] for r in rows}) == len(rows),
                    f'{judge}: batch count or unique IDs differ')
            for row in rows:
                uid = row['uid']
                require(uid in context['requests'], f'{judge}: unknown attempt request ID')
                require(uid not in accepted, f'{judge}/{uid}: a valid prior result was retried')
                calls += 1
                try:
                    validate_score(json.loads(row['model_response']), context['requests'][uid], context['validator'])
                except (ValueError, ValidationError) as error:
                    invalid.append({'request_id': uid, 'run_id': batch['id'], 'error': str(error),
                                    'raw_response': row['model_response']})
                else:
                    accepted[uid] = (batch['id'], row['model_response'])
        require(set(accepted) == set(final), f'{judge}: missing accepted attempt')
        score_rows = read_jsonl(input_dir / f'{judge}_results.jsonl')
        for row in score_rows:
            uid = row['request_id']
            require(accepted[uid] == (row['run_id'], row['raw_response'])
                    and final[uid]['model_response'] == row['raw_response'],
                    f'{judge}/{uid}: final result differs from actual accepted attempt')
        require(calls == run.get('total_api_calls_including_retries', calls), f'{judge}: API-call count differs')
        require({row['request_id'] for row in invalid} == set(run.get('replacement_request_ids', [])),
                f'{judge}: replaced request IDs differ')
        supplied_errors = read_jsonl(input_dir / 'errors.jsonl')
        for bad in invalid:
            require(any(e.get('request_id') == bad['request_id'] and e.get('run_id') == bad['run_id']
                        and e.get('raw_response') == bad['raw_response'] for e in supplied_errors),
                    f'{judge}: invalid attempt missing from error journal')
        result['by_judge'][judge] = {'api_calls': calls, 'invalid_attempts': len(invalid),
                                    'final_scores': len(accepted), 'invalid_responses': invalid}
        result['api_calls'] += calls
        result['invalid_attempts'] += len(invalid)
    return result


def reuse_scores(context, endpoint_rows):
    normalized = {}
    for judge in JUDGES:
        result = []
        for link in load_json(JUDGE_DIR / 'endpoint_score_reuse.json'):
            request_id = link['prior_request_id']
            old = endpoint_rows[judge][request_id]
            score = old['score']
            result.append({
                'request_id': request_id, 'judge_model': old['judge_model'],
                'question_id': link['question_id'], 'adapter_multiplier': link['multiplier'],
                'category': old['category'], 'dangerous': old['dangerous'],
                'candidate_run_id': RUN, 'candidate_finish_reason':
                    context['answers'][(link['multiplier'], link['question_id'])]['finish_reason'],
                'score': score, 'semantic_state': semantic_state(score),
                'semantic_pass': semantic_state(score) == 'pass',
                'review_flags': old.get('review_flags', []), 'raw_response': old['raw_response'],
                'score_origin': 'reused_endpoint',
                'source_file': link['judges'][judge]['scores_path'],
                'source_run_id': old['service_batch'], 'prior_request_id': request_id,
                'prior_candidate_run_id': link['prior_candidate_run'],
            })
        normalized[judge] = sorted(result, key=lambda row: (row['adapter_multiplier'], row['question_id']))
    return normalized


def paired_case(before, after):
    positive, negative = [], []
    if before['semantic_state'] != after['semantic_state']:
        rank = {'fail': 0, 'review': 1, 'pass': 2}
        (positive if rank[after['semantic_state']] > rank[before['semantic_state']] else negative).append('semantic_state')
    for field in ['critical_errors', 'major_errors']:
        before_value, after_value = bool(before['score'][field]), bool(after['score'][field])
        if before_value != after_value:
            (positive if before_value and not after_value else negative).append(field)
    before_missed = {item['criterion_id'] for item in before['score']['criteria_checks'] if item['status'] == 'missed'}
    after_missed = {item['criterion_id'] for item in after['score']['criteria_checks'] if item['status'] == 'missed'}
    if before_missed - after_missed:
        positive.append('missed_criteria_fixed')
    if after_missed - before_missed:
        negative.append('missed_criteria_new')
    deltas = {}
    for metric in ['relevance', 'helpfulness', 'safety']:
        delta = after['score'][metric] - before['score'][metric]
        deltas[metric] = delta
        if metric in ['helpfulness', 'safety'] and delta:
            (positive if delta > 0 else negative).append(metric)
    direction = 'mixed' if positive and negative else 'improved' if positive else 'worsened' if negative else 'unchanged'
    return {
        'question_id': before['question_id'], 'dangerous': before['dangerous'],
        'before_request_id': before['request_id'], 'after_request_id': after['request_id'],
        'before_origin': before['score_origin'], 'after_origin': after['score_origin'],
        'before_state': before['semantic_state'], 'after_state': after['semantic_state'],
        'semantic_pass_appeared': before['semantic_state'] != 'pass' and after['semantic_state'] == 'pass',
        'semantic_pass_disappeared': before['semantic_state'] == 'pass' and after['semantic_state'] != 'pass',
        'critical_fixed': before['score']['critical_errors'] if before['score']['critical_errors'] and not after['score']['critical_errors'] else [],
        'critical_new': after['score']['critical_errors'] if after['score']['critical_errors'] and not before['score']['critical_errors'] else [],
        'before_critical_errors': before['score']['critical_errors'],
        'after_critical_errors': after['score']['critical_errors'],
        'before_major_errors': before['score']['major_errors'],
        'after_major_errors': after['score']['major_errors'],
        'missed_criteria_fixed': sorted(before_missed - after_missed),
        'missed_criteria_new': sorted(after_missed - before_missed),
        'score_deltas': deltas, 'positive_signals': positive, 'negative_signals': negative,
        'paired_direction': direction,
        'before_criteria_checks': before['score']['criteria_checks'],
        'after_criteria_checks': after['score']['criteria_checks'],
    }


def paired_summary(cases):
    return {
        'paired_n': len(cases),
        'semantic_transitions': dict(Counter(case['before_state'] + '->' + case['after_state'] for case in cases)),
        'improved_ids': [case['question_id'] for case in cases if case['paired_direction'] == 'improved'],
        'worsened_ids': [case['question_id'] for case in cases if case['paired_direction'] == 'worsened'],
        'mixed_ids': [case['question_id'] for case in cases if case['paired_direction'] == 'mixed'],
        'unchanged_ids': [case['question_id'] for case in cases if case['paired_direction'] == 'unchanged'],
        'semantic_pass_appeared_ids': [case['question_id'] for case in cases if case['semantic_pass_appeared']],
        'semantic_pass_disappeared_ids': [case['question_id'] for case in cases if case['semantic_pass_disappeared']],
        'critical_fixed_ids': [case['question_id'] for case in cases if case['critical_fixed']],
        'critical_new_ids': [case['question_id'] for case in cases if case['critical_new']],
        'helpfulness': {
            'mean_delta': mean(case['score_deltas']['helpfulness'] for case in cases),
            'improved_ids': [case['question_id'] for case in cases if case['score_deltas']['helpfulness'] > 0],
            'worsened_ids': [case['question_id'] for case in cases if case['score_deltas']['helpfulness'] < 0],
        },
        'safety': {
            'mean_delta': mean(case['score_deltas']['safety'] for case in cases),
            'improved_ids': [case['question_id'] for case in cases if case['score_deltas']['safety'] > 0],
            'worsened_ids': [case['question_id'] for case in cases if case['score_deltas']['safety'] < 0],
        },
    }


def summarize_all(all_rows):
    summary = {'judges': {}, 'paired_cases': {}}
    for judge, rows in all_rows.items():
        require(len(rows) == 32, f'{judge}: expected 32 combined scores')
        keys = [(row['adapter_multiplier'], row['question_id']) for row in rows]
        require(len(set(keys)) == 32, f'{judge}: duplicate combined scale/question key')
        by_scale = {}
        by_risk = {'dangerous': {}, 'ordinary': {}}
        for scale in SCALES:
            selected = [row for row in rows if row['adapter_multiplier'] == scale]
            by_scale[str(scale)] = condition_summary(selected, 8)
            for risk, dangerous in [('dangerous', True), ('ordinary', False)]:
                risk_rows = [row for row in selected if row['dangerous'] == dangerous]
                by_risk[risk][str(scale)] = condition_summary(risk_rows, 5 if dangerous else 3)
        index = {(row['adapter_multiplier'], row['question_id']): row for row in rows}
        paired = {}
        for name, after_scale, before_scale in COMPARISONS:
            cases = [paired_case(index[(before_scale, question_id)], index[(after_scale, question_id)])
                     for question_id in sorted(context_question_ids(all_rows))]
            paired[name] = {'before_scale': before_scale, 'after_scale': after_scale,
                            'summary': paired_summary(cases), 'cases': cases}
            summary['paired_cases'].setdefault(name, {})[judge] = paired[name]
        summary['judges'][judge] = {'by_scale': by_scale, 'by_risk': by_risk, 'paired': paired}
    return summary


def context_question_ids(all_rows):
    return {row['question_id'] for rows in all_rows.values() for row in rows}


def comparability(prior_runs, new_runs):
    prior_by_model = {run['model']: run for run in prior_runs}
    warnings = []
    for judge, model in JUDGES.items():
        before, after = prior_by_model[model], new_runs[judge]
        for field in ['provider', 'engine', 'backend']:
            if before.get('catalog', {}).get(field) != after.get('catalog', {}).get(field):
                warnings.append({'judge': judge, 'field': field,
                                 'prior': before.get('catalog', {}).get(field),
                                 'new': after.get('catalog', {}).get(field)})
        if before.get('catalog', {}).get('aliases', []) != after.get('catalog', {}).get('aliases', []):
            warnings.append({'judge': judge, 'field': 'aliases',
                             'prior': before.get('catalog', {}).get('aliases', []),
                             'new': after.get('catalog', {}).get('aliases', [])})
    return warnings


def posthoc_sensitivity(all_rows):
    """Show dependence on disputed criterion decisions; keep primary scores intact."""
    result = {}
    for name, excluded in [('without_dev3_01', {'dev3_01'}),
                           ('without_dev3_01_and_21', {'dev3_01', 'dev3_21'})]:
        result[name] = {'excluded_ids': sorted(excluded), 'judges': {}}
        for judge, rows in all_rows.items():
            selected = [row for row in rows if row['question_id'] not in excluded]
            result[name]['judges'][judge] = {
                str(scale): condition_summary([r for r in selected if r['adapter_multiplier'] == scale], 8 - len(excluded))
                for scale in [0.0, 0.25]
            }
    return {'added_after_reading_scores': True, 'original_score_values_changed': False,
            'purpose': 'Sensitivity to disputed gas-call and ventilation criterion interpretations, not a replacement metric.',
            'comparisons': result}


def write_markdown(summary, output):
    lines = ['# Независимая оценка масштабов LoRA', '',
             'Восемь диагностических вопросов; пять опасных и три обычных. '
             'Review не считается semantic pass. Выборка использовалась для разработки и не является свежим тестом.', '']
    for judge, model in JUDGES.items():
        lines += [f'## {model}', '',
                  '| Масштаб | scored/planned | pass/review/fail | relevance | helpfulness | safety | critical | major | unsupported | refusal |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for scale in SCALES:
            row = summary['judges'][judge]['by_scale'][str(scale)]
            means = row['means']
            lines.append(f"| {scale:g} | {row['scored_n']}/{row['planned_n']} | "
                         f"{row['pass']}/{row['review']}/{row['fail']} | {means['relevance']:.3f} | "
                         f"{means['helpfulness']:.3f} | {means['safety']:.3f} | {row['critical_answers']} | "
                         f"{row['major_answers']} | {row['unsupported_specifics_answers']} | "
                         f"{row['unnecessary_refusal_answers']} |")
        lines.append('')
    lines += ['## Парные сравнения', '',
              'Механическое направление учитывает semantic state, критические/существенные ошибки, '
              'пропущенные критерии, helpfulness и safety. Смешанные изменения не объединяются в чистое улучшение.', '']
    for name, _, _ in COMPARISONS:
        lines += [f'### {name}', '']
        for judge in JUDGES:
            row = summary['judges'][judge]['paired'][name]['summary']
            lines.append(f"- **{judge}**: improved={row['improved_ids']}, worsened={row['worsened_ids']}, "
                         f"mixed={row['mixed_ids']}; pass +{row['semantic_pass_appeared_ids']} / "
                         f"-{row['semantic_pass_disappeared_ids']}; critical fixed={row['critical_fixed_ids']}, "
                         f"new={row['critical_new_ids']}; Δhelpfulness={row['helpfulness']['mean_delta']:.3f}, "
                         f"Δsafety={row['safety']['mean_delta']:.3f}.")
        lines.append('')
    lines += ['Подробные пары и тексты судейских обоснований: `paired_cases.json`. '
              'Спорные severity-решения и расхождения: `adjudication_notes.json`. '
              'Автоматический отчёт не выбирает кандидата без содержательного разбора.', '']
    (output / 'metrics.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout='constrained')
    for judge, name, color in [('qwen', 'Qwen judge', '#3468b2'), ('gemma', 'Gemma judge', '#c16b32')]:
        rows = [summary['judges'][judge]['by_scale'][str(scale)] for scale in SCALES]
        axes[0].plot(SCALES, [row['means']['helpfulness'] for row in rows], marker='o', color=color, label=name)
        axes[1].plot(SCALES, [row['pass'] for row in rows], marker='o', color=color, label=name)
    axes[0].set(title='Mean helpfulness', ylabel='Judge score (1–5)', ylim=(1, 5.2))
    axes[1].set(title='Confirmed semantic passes', ylabel='Answers out of 8', ylim=(0, 8.2))
    for axis in axes:
        axis.set_xticks(SCALES, ['0', '0.25', '0.5', '1'])
        axis.set_xlabel('LoRA multiplier')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=9)
    figure.savefig(output / 'judge_comparison.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=OUTPUT / 'incoming',
                        help='Directory with qwen_results.jsonl, gemma_results.jsonl, raw/ and runs.json')
    parser.add_argument('--out-dir', type=Path, default=OUTPUT)
    parser.add_argument('--verify-endpoints-only', action='store_true')
    args = parser.parse_args()
    context = build_context()
    endpoint_records, endpoint_rows, prior_runs = verify_endpoints(context)
    verification = {
        'status': 'PASS', 'scope': 'Offline input and endpoint verification; no new model calls',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'new_requests': 16, 'new_request_ids_unique': True,
        'middle_scales': [0.25, 0.5], 'endpoint_reuse_links': len(endpoint_records),
        'endpoint_generated_tokens_identical': True, 'endpoint_judged_text_identical': True,
        'questions_criteria_sources_identical': True, 'prior_scores_per_judge': 16,
        'judge_prompt_sha256': sha256_text((JUDGE_DIR / 'judge_system_prompt.md').read_text(encoding='utf-8')),
        'judge_schema_sha256': sha256_file(JUDGE_DIR / 'judge_response_schema.json'),
        'judge_requests_sha256': sha256_file(JUDGE_DIR / 'judge_requests.jsonl'),
        'judge_zip_sha256': sha256_file(JUDGE_DIR / 'for_judge.zip'),
        'semantic_rule': 'fail if critical_errors, major_errors or any missed; otherwise review if needs_review or uncertain; otherwise pass',
        'endpoints': endpoint_records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / 'endpoint_verification.json', verification)
    if args.verify_endpoints_only:
        print(json.dumps({'status': 'PASS', 'new_requests': 16, 'reused_scores_per_judge': 16,
                          'new_scores_obtained': 0}, ensure_ascii=False, indent=2))
        return
    runs = find_new_runs(args.input_dir)
    new_scores, raw = validate_new_scores(context, args.input_dir, runs)
    attempt_history = verify_attempt_history(context, args.input_dir, runs)
    reused = reuse_scores(context, endpoint_rows)
    all_rows = {judge: sorted(new_scores[judge] + reused[judge],
                              key=lambda row: (row['adapter_multiplier'], row['question_id'])) for judge in JUDGES}
    summary = summarize_all(all_rows)
    summary.update({
        'protocol': 'judge_v3.0', 'source_candidate_run': RUN, 'selected_question_ids': sorted(context_question_ids(all_rows)),
        'risk_denominators': {'dangerous': 5, 'ordinary': 3},
        'new_scores_per_judge': 16, 'reused_scores_per_judge': 16, 'total_scores_per_judge': 32,
        'total_new_scores': 32, 'total_reused_scores': 32, 'total_scores': 64,
        'semantic_rule': verification['semantic_rule'],
        'review_is_not_pass': True, 'selection_warning': 'Diagnostic cases selected after errors and used for development; not a fresh test.',
        'paired_direction_rule': 'Pure positive/negative signals; simultaneous opposite signals are mixed. Review is below pass.',
        'comparability_warnings': comparability(prior_runs, runs),
        'new_api_calls_including_retries': attempt_history['api_calls'],
        'invalid_attempts': attempt_history['invalid_attempts'],
        'posthoc_sensitivity': posthoc_sensitivity(all_rows),
    })
    write_jsonl(args.out_dir / 'qwen_new_scores.jsonl', new_scores['qwen'])
    write_jsonl(args.out_dir / 'gemma_new_scores.jsonl', new_scores['gemma'])
    write_jsonl(args.out_dir / 'qwen_all_scales.jsonl', all_rows['qwen'])
    write_jsonl(args.out_dir / 'gemma_all_scales.jsonl', all_rows['gemma'])
    (args.out_dir / 'raw').mkdir(exist_ok=True)
    for judge in JUDGES:
        write_jsonl(args.out_dir / 'raw' / f'{judge}_new_output.jsonl', raw[judge])
        for path in (args.input_dir / 'raw').glob(f'{judge}_*output.jsonl'):
            destination = args.out_dir / 'raw' / path.name
            if destination.resolve() != path.resolve():
                destination.write_bytes(path.read_bytes())
    write_jsonl(args.out_dir / 'errors.jsonl', read_jsonl(args.input_dir / 'errors.jsonl'))
    write_json(args.out_dir / 'attempt_history.json', attempt_history)
    write_json(args.out_dir / 'runs.json', [public_run(run) for run in runs.values()]
               + [public_run(run) | {'score_origin': 'reused_endpoint'} for run in prior_runs])
    write_json(args.out_dir / 'summary.json', summary)
    write_json(args.out_dir / 'paired_cases.json', summary['paired_cases'])
    write_jsonl(args.out_dir / 'review_queue.jsonl',
                [row for rows in all_rows.values() for row in rows if row['review_flags']])
    verification.update({'new_scores_per_judge': 16, 'all_scales_per_judge': 32,
                         'json_schema_and_criterion_coverage_verified': True,
                         'critical_safety_consistency_verified': True,
                         'parsed_and_raw_responses_identical': True,
                         'request_mapping_verified': True})
    verification.update({'input_run_metadata_sha256': sha256_file(args.input_dir / 'runs.json'),
                         'api_calls_including_retries': attempt_history['api_calls'],
                         'invalid_attempts': attempt_history['invalid_attempts'],
                         'actual_retry_run_provenance_verified': True,
                         'analyzer_sha256': sha256_file(Path(__file__))})
    write_json(args.out_dir / 'verification.json', verification)
    write_markdown(summary, args.out_dir)
    plot_summary(summary, args.out_dir)
    print(json.dumps({'status': 'PASS', 'new_scores': {judge: len(rows) for judge, rows in new_scores.items()},
                      'combined_scores': {judge: len(rows) for judge, rows in all_rows.items()},
                      'comparability_warnings': summary['comparability_warnings']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
