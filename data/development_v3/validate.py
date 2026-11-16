"""Validate the frozen development set with Python's standard library."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
CATEGORIES = {
    'gas', 'electricity', 'water', 'structural',
    'doors_windows', 'finishes', 'appliances', 'housing',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    require(all(line.strip() for line in lines), f'Empty line in {path.name}')
    return [json.loads(line) for line in lines]


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    return ' '.join(re.findall(r'\w+', text))


def validate():
    questions = read_jsonl(DIRECTORY / 'questions.jsonl')
    rubrics = read_jsonl(DIRECTORY / 'rubric.jsonl')
    registry = json.loads((DIRECTORY / 'sources.json').read_text(encoding='utf-8'))
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text(encoding='utf-8'))
    sources = registry['sources']
    ids = [f'dev3_{number:02d}' for number in range(1, 33)]
    require([q['id'] for q in questions] == ids, 'Question IDs or order differ')
    require([r['id'] for r in rubrics] == ids, 'Rubric IDs or order differ')
    require(len({normalize(q['question']) for q in questions}) == 32, 'Duplicate questions')
    counts = Counter(q['category'] for q in questions)
    require(counts == Counter({category: 4 for category in CATEGORIES}), 'Category balance differs')
    require(dict(counts) == manifest['category_counts'], 'Manifest category counts differ')

    for question, rubric in zip(questions, rubrics):
        case_id = question['id']
        require(set(question) == {'id', 'category', 'dangerous', 'question'}, f'Unexpected input fields in {case_id}')
        require(type(question['dangerous']) is bool, f'Invalid danger flag in {case_id}')
        require(bool(question['question'].strip()), f'Empty question in {case_id}')
        criteria = rubric['required_criteria']
        require(2 <= len(criteria) <= 3, f'Invalid criterion count in {case_id}')
        require([c['id'] for c in criteria] == [f'R{i}' for i in range(1, len(criteria) + 1)], f'Invalid criterion IDs in {case_id}')
        for criterion in criteria:
            require(bool(criterion['text'].strip()), f'Empty criterion in {case_id}')
            require(bool(criterion['source_ids']), f'Missing basis in {case_id}')
            require(set(criterion['source_ids']) <= sources.keys(), f'Unknown source in {case_id}')
        require(rubric['information_status'] in {'sufficient', 'conditional', 'clarification_needed'}, f'Invalid information status in {case_id}')
        if rubric['information_status'] != 'sufficient':
            require(bool(rubric['missing_information']), f'Missing clarification needs in {case_id}')
        if question['dangerous']:
            require(bool(rubric['critical_errors']), f'Missing critical errors in {case_id}')
        require(bool(rubric['major_errors']), f'Missing major errors in {case_id}')
        require(bool(rubric['acceptable_variants']), f'Missing acceptable variants in {case_id}')
        require(bool(rubric['grounding_note']), f'Missing grounding note in {case_id}')

    for category in CATEGORIES:
        require(sum(q['dangerous'] for q in questions if q['category'] == category) == 2, f'Danger balance differs in {category}')
    require(manifest['dangerous_counts'] == {'dangerous': 16, 'ordinary': 16}, 'Manifest danger counts differ')
    require(manifest['task_type_counts'] == dict(Counter(r['task_type'] for r in rubrics)), 'Task counts differ')
    require(manifest['information_status_counts'] == dict(Counter(r['information_status'] for r in rubrics)), 'Information counts differ')
    external = [source for source in sources.values() if source['url']]
    require(len(external) == manifest['external_sources'] == 17, 'Source count differs')
    require(all(source['verification_status'] == 'content_read' for source in external), 'External source verification is incomplete')
    require(all(source['limits'] and source['supports'] for source in sources.values()), 'Incomplete source scope')

    expected_files = {'questions.jsonl', 'rubric.jsonl', 'sources.json', 'README.md', 'validate.py'}
    require(set(manifest['file_sha256']) == expected_files, 'Incomplete file hash list')
    for name, expected_hash in manifest['file_sha256'].items():
        actual_hash = hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
        require(actual_hash == expected_hash, f'File changed since freeze: {name}')

    current_questions = {normalize(q['question']) for q in questions}
    checked_files = []
    changed_reference_files = []
    for reference in manifest['duplicate_checks']:
        path = ROOT / reference['path']
        records = read_jsonl(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            changed_reference_files.append(reference['path'])
        texts = []
        for record in records:
            if 'question' in record:
                texts.append(record['question'])
            else:
                texts.extend(message['content'] for message in record['messages'] if message['role'] == 'user')
        matches = current_questions.intersection(normalize(text) for text in texts)
        require(not matches, f'Development questions overlap exactly with {path}')
        checked_files.append(reference['path'])

    require(math.isclose(.008 * 5 * 30, 1.2), 'Lamp calculation failed')
    require(3427 - 3418 == 9 and 9 * 50 == 450, 'Water calculation failed')
    require(3 * 26 == 78 and 78 <= 80 and 30 <= 32 and 25 <= 28, 'Box calculation failed')
    require(math.ceil(2.6 / .53) == 5, 'Wallpaper calculation failed')
    require(manifest['arithmetic_checks'] == {
        'lamp_kwh': 1.2, 'water_m3': 9, 'water_cost_rub': 450,
        'boxes_total_width_cm': 78, 'boxes_fit': True, 'wallpaper_strips': 5,
    }, 'Manifest arithmetic differs')

    return {
        'status': 'OK', 'questions': len(questions), 'rubrics': len(rubrics),
        'categories': dict(counts), 'dangerous': 16, 'ordinary': 16,
        'external_sources': len(external), 'hashed_files': len(expected_files),
        'reference_files_checked': len(checked_files), 'normalized_exact_matches': 0,
        'changed_reference_files_since_freeze': changed_reference_files,
        'semantic_scenario_overlap': 'present, see README.md',
        'expert_human_review': False,
    }


if __name__ == '__main__':
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
