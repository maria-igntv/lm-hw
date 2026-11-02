"""Pair saved answers by question ID; never compare different question sets."""

from collections import Counter

from checks.autocheck import check_answer, check_dataset


def index_records(records):
    indexed = {}
    for record in records:
        key = record['id']
        if key in indexed:
            raise ValueError(f'Duplicate question ID: {key}')
        indexed[key] = record
    return indexed


def validate_answers(questions, answers, *, complete=True):
    expected = index_records(questions)
    actual = index_records(answers)
    if not set(actual) <= set(expected):
        raise ValueError('Answers contain unknown question IDs')
    if complete and set(actual) != set(expected):
        raise ValueError('Answers do not cover the complete evaluation set')
    for key, record in actual.items():
        for field in ('question', 'dangerous', 'category'):
            if record.get(field) != expected[key][field]:
                raise ValueError(f'ID {key}: different {field}')
        if not isinstance(record.get('answer'), str) or not record['answer'].strip():
            raise ValueError(f'ID {key}: empty or invalid answer')
    return actual


def compare_answers(questions, baseline, sft):
    before = validate_answers(questions, baseline)
    after = validate_answers(questions, sft)
    transitions = Counter()
    details = []
    for question in questions:
        key = question['id']
        left, right = before[key], after[key]
        b = check_answer(question['question'], left['answer'], question['dangerous'])
        a = check_answer(question['question'], right['answer'], question['dangerous'])
        transition = 'not_applicable'
        if question['dangerous']:
            transition = {
                (False, True): 'fixed', (True, False): 'regressed',
                (False, False): 'still_missing', (True, True): 'both_pass',
            }[(b.safety_pass, a.safety_pass)]
            transitions[transition] += 1
        details.append({
            **question, 'baseline_answer': left['answer'], 'sft_answer': right['answer'],
            'baseline_safety_pass': b.safety_pass, 'sft_safety_pass': a.safety_pass,
            'transition': transition, 'baseline_failures': b.failures,
            'sft_failures': a.failures,
        })
    summaries = {}
    for name, records in [('baseline', baseline), ('sft', sft)]:
        summaries[name] = {k: v for k, v in check_dataset(records).items() if k != 'failures'}
    return summaries, dict(transitions), details
