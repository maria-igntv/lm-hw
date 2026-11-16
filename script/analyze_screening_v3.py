"""Verify Colab screening archives and assemble a mixed, blinded judge batch."""

import argparse
import ast
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.screening_v3 import read_jsonl, sha256_file, summarize, validate_records, write_json, write_jsonl


def read_archive(path, questions, protocol):
    with zipfile.ZipFile(path) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith('/run_metadata.json')]
        if len(metadata_names) != 1:
            raise ValueError(f'{path.name}: expected one run metadata file')
        prefix = metadata_names[0].removesuffix('run_metadata.json')
        contents = {}
        for item in archive.infolist():
            if item.is_dir():
                continue
            relative = PurePosixPath(item.filename).relative_to(prefix.rstrip('/'))
            if '..' in relative.parts or relative.is_absolute():
                raise ValueError(f'Invalid archive member {item.filename}')
            contents[str(relative)] = archive.read(item)

    metadata = json.loads(contents['run_metadata.json'])
    signature = metadata['signature']
    signature_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf-8')).hexdigest()
    if signature_hash != metadata['signature_sha256']:
        raise ValueError(f'{path.name}: invalid signature hash')
    key = signature['model_key']
    if signature['candidate'] != protocol['models'][key]:
        raise ValueError(f'{path.name}: model revision differs')
    expected_run_id = f"{key}_t{protocol['generation']['max_new_tokens']}_{signature_hash[:12]}"
    if metadata['run_id'] != expected_run_id or prefix.rstrip('/') != expected_run_id:
        raise ValueError(f'{path.name}: run ID differs from signature')
    for filename, expected in signature['file_sha256'].items():
        if hashlib.sha256(contents['reproduction/' + filename]).hexdigest() != expected:
            raise ValueError(f'{path.name}: changed archived input {filename}')
        if sha256_file(ROOT / filename) != expected:
            raise ValueError(f'{path.name}: current input differs from archived input {filename}')

    records = [json.loads(line) for line in contents['answers.jsonl'].decode('utf-8').splitlines() if line.strip()]
    indexed = validate_records(questions, protocol['prompts'], records, complete=True)
    actual_summary = summarize(questions, protocol['prompts'], records)
    if actual_summary != json.loads(contents['automatic_summary.json']):
        raise ValueError(f'{path.name}: automatic summary differs')
    if metadata.get('status') != 'complete' or not metadata.get('complete'):
        raise ValueError(f'{path.name}: incomplete run')
    technical = json.loads(contents['technical_smoke.json'])
    if technical['status'] != 'complete' or not technical['last_logits_finite']:
        raise ValueError(f'{path.name}: incomplete numerical controls')
    effective = json.loads(contents['effective_generation_config.json'])
    for setting, value in protocol['generation'].items():
        if effective.get(setting) != value:
            raise ValueError(f'{path.name}: generation setting differs: {setting}')
    rubric_index = {record['id']: record for record in read_jsonl(ROOT / 'data/development_v3/rubric.jsonl')}
    expected_system = (ROOT / protocol['judge']['system_prompt_path']).read_text(encoding='utf-8')
    if contents['judge_system_prompt.md'].decode('utf-8') != expected_system:
        raise ValueError(f'{path.name}: judge prompt differs from protocol')
    for record in records:
        if record['run_id'] != metadata['run_id'] or record['model_key'] != key:
            raise ValueError(f'{path.name}: answer belongs to another run')
        if record['model_revision'] != signature['candidate']['revision']:
            raise ValueError(f'{path.name}: answer revision differs')

    requests = [json.loads(line) for line in contents['judge_requests.jsonl'].decode('utf-8').splitlines() if line.strip()]
    mapping = json.loads(contents['judge_mapping.json'])
    request_ids = [request['request_id'] for request in requests]
    mapped = {item['request_id']: item for item in mapping}
    if len(request_ids) != len(indexed) or len(set(request_ids)) != len(indexed) or set(mapped) != set(request_ids):
        raise ValueError(f'{path.name}: inconsistent judge request IDs')
    for request in requests:
        link = mapped[request['request_id']]
        record = indexed[(link['prompt_id'], link['question_id'])]
        payload = json.loads(request['user_prompt'])
        if payload['assistant_answer'] != record['answer'] or payload['question'] != record['question']:
            raise ValueError(f'{path.name}: judge input differs from saved answer')
        if payload['case_criteria'] != rubric_index[record['id']]:
            raise ValueError(f'{path.name}: judge criteria differ')
        if request['system_prompt'] != contents['judge_system_prompt.md'].decode('utf-8'):
            raise ValueError(f'{path.name}: judge system prompt differs')

    return dict(path=path, contents=contents, metadata=metadata, records=records,
                requests=requests, mapping=mapping, summary=actual_summary)


def notebook_summary(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    original = json.loads((ROOT / 'notebooks/00_model_screening_v3.ipynb').read_text(encoding='utf-8'))
    actual_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    original_cells = [cell for cell in original['cells'] if cell['cell_type'] == 'code']
    if len(actual_cells) != len(original_cells):
        raise ValueError(f'{path.name}: code cell count differs')
    differences = []
    model_key = None
    for number, (actual, expected) in enumerate(zip(actual_cells, original_cells), 1):
        actual_tree = ast.parse(''.join(actual['source']))
        expected_tree = ast.parse(''.join(expected['source']))
        for tree in (actual_tree, expected_tree):
            assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'MODEL_KEY' for target in node.targets)]
            for assignment in assignments:
                if tree is actual_tree:
                    model_key = ast.literal_eval(assignment.value)
                tree.body.remove(assignment)
        if ast.dump(actual_tree) != ast.dump(expected_tree):
            differences.append(number)
    errors = [
        {'cell': number, 'name': output.get('ename'), 'message': output.get('evalue')}
        for number, cell in enumerate(notebook['cells'])
        for output in cell.get('outputs', []) if output['output_type'] == 'error'
    ]
    return dict(filename=path.name, sha256=sha256_file(path), model_key=model_key,
                code_cells=len(actual_cells), execution_counts=[cell.get('execution_count') for cell in actual_cells],
                changed_code_cells_except_model_key=differences, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs=2, type=Path)
    parser.add_argument('--notebook', action='append', type=Path, default=[])
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'data/model_screening_v3/protocol.json').read_text(encoding='utf-8'))
    questions = read_jsonl(ROOT / 'data/development_v3/questions.jsonl')
    runs = [read_archive(path, questions, protocol) for path in args.archives]
    by_model = {run['metadata']['signature']['model_key']: run for run in runs}
    if set(by_model) != set(protocol['models']):
        raise ValueError('Expected one archive per candidate')
    left, right = [run['metadata']['signature'] for run in runs]
    for field in ('protocol_id', 'file_sha256', 'versions', 'hardware'):
        if left[field] != right[field]:
            raise ValueError(f'Conditions differ between runs: {field}')
    notebook_checks = [notebook_summary(path) for path in args.notebook]

    results_dir = ROOT / 'results/model_screening_v3'
    results_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    mapping = []
    metrics = []
    provenance = []
    for model_key in protocol['models']:
        run = by_model[model_key]
        run_dir = results_dir / run['metadata']['run_id']
        run_dir.mkdir(parents=True, exist_ok=True)
        for relative, payload in run['contents'].items():
            destination = run_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError(f'Existing artifact differs: {destination}')
            destination.write_bytes(payload)
        requests.extend(run['requests'])
        mapping.extend({**item, 'model_key': model_key} for item in run['mapping'])
        provenance.append(dict(archive=run['path'].name, archive_sha256=sha256_file(run['path']),
                               run_id=run['metadata']['run_id'], answers_sha256=sha256_file(run_dir / 'answers.jsonl')))
        for prompt_id, summary in run['summary']['by_prompt'].items():
            metrics.append({
                'model_key': model_key, 'prompt_id': prompt_id, **summary,
                'warning_marker_fraction': summary['dangerous_with_warning_marker'] / summary['dangerous_n'],
                'length_stop_fraction': summary['length_stops'] / summary['n'],
                'generation_minutes': summary['total_generation_seconds'] / 60,
            })

    for filename in ('judge_system_prompt.md', 'judge_response_schema.json', 'judge_run_config.json'):
        if runs[0]['contents'][filename] != runs[1]['contents'][filename]:
            raise ValueError(f'Judge settings differ: {filename}')
    if len({request['request_id'] for request in requests}) != 128:
        raise ValueError('Expected 128 unique judge requests')
    random.Random(protocol['seed']).shuffle(requests)
    judge_dir = ROOT / 'judge/model_screening_v3'
    judge_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(judge_dir / 'judge_requests.jsonl', requests)
    write_json(judge_dir / 'judge_mapping.json', mapping)
    for filename in ('judge_system_prompt.md', 'judge_response_schema.json'):
        (judge_dir / filename).write_bytes(runs[0]['contents'][filename])
    judge_config = json.loads(runs[0]['contents']['judge_run_config.json'])
    judge_config.update(expected_requests=128, exported_requests=128,
                        system_prompt_path='judge_system_prompt.md', response_schema_path='judge_response_schema.json')
    write_json(judge_dir / 'judge_run_config.json', judge_config)
    write_json(judge_dir / 'manifest.json', {
        'protocol_id': protocol['protocol_id'], 'judge_protocol_id': 'judge_v3.0',
        'requests': 128, 'questions': 32, 'conditions': 4,
        'judgments_obtained': 0, 'shuffle_seed': protocol['seed'], 'source_archives': provenance,
        'counts': dict(Counter(item['model_key'] + '/' + item['prompt_id'] for item in mapping)),
        'request_file_sha256': sha256_file(judge_dir / 'judge_requests.jsonl'),
    })
    bundle_files = ['judge_requests.jsonl', 'judge_system_prompt.md', 'judge_response_schema.json', 'judge_run_config.json']
    with zipfile.ZipFile(judge_dir / 'for_judge.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename in bundle_files:
            archive.write(judge_dir / filename, arcname=filename)

    truncated = [
        {'model_key': key, 'prompt_id': record['prompt_id'], 'question_id': record['id'],
         'category': record['category'], 'dangerous': record['dangerous']}
        for key, run in by_model.items() for record in run['records'] if record['finish_reason'] == 'length'
    ]
    write_json(results_dir / 'comparison.json', {
        'protocol_id': protocol['protocol_id'], 'total_answers': 128, 'metrics': metrics,
        'truncated_answers': truncated, 'source_archives': provenance, 'notebook_checks': notebook_checks,
        'matched_fields': ['protocol_id', 'file_sha256', 'versions', 'hardware'],
        'judge_status': 'prepared_128_requests',
    })
    table = [
        '# Автоматическое сравнение screening v3\n\n',
        '| Модель | Промпт | Ответов | Маркер на опасных | Обрыв по лимиту | CJK | Повторы | Генерация, мин |\n',
        '| --- | --- | --- | --- | --- | --- | --- | --- |\n',
    ]
    for metric in metrics:
        table.append(f"| {metric['model_key']} | {metric['prompt_id']} | {metric['n']} | "
                     f"{metric['dangerous_with_warning_marker']}/{metric['dangerous_n']} | "
                     f"{metric['length_stops']}/{metric['n']} | {metric['answers_with_cjk']} | "
                     f"{metric['answers_with_repetition']} | {metric['generation_minutes']:.2f} |\n")
    table.append('\nЧастота маркера отражает работу regex. Содержательные выводы приведены в analysis.md.\n')
    (results_dir / 'comparison.md').write_text(''.join(table), encoding='utf-8')
    print(json.dumps({'verified_runs': len(runs), 'answers': 128, 'judge_requests': 128,
                      'truncated_answers': len(truncated), 'results': str(results_dir),
                      'judge_bundle': str(judge_dir / 'for_judge.zip')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
