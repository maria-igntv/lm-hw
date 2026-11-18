"""Verify a completed SFT v3 pilot archive and preserve its paired results."""

import argparse
import ast
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from safetensors.numpy import load as load_safetensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import sha256_file, summarize, technical_indicators, write_json
from checks.sft_pilot_v3 import select_pilot_records, validate_pilot_records


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def code_asts(notebook):
    return [ast.dump(ast.parse(''.join(cell['source'])), include_attributes=False)
            for cell in notebook['cells'] if cell['cell_type'] == 'code']


def repeated_span(answer):
    """Post-hoc diagnostic: three exact adjacent copies, period 3..128 words.

    The frozen autocheck uses 3..12. This extra diagnostic does not replace
    that score or change archived answers. It was added after inspecting loops.
    """
    words = re.findall(r'\w+', answer.casefold())
    for width in range(3, min(128, len(words) // 3) + 1):
        for start in range(len(words) - width * 3 + 1):
            chunk = words[start:start + width]
            if words[start + width:start + width * 3] == chunk * 2:
                return {'period_words': width, 'start_word': start, 'block': ' '.join(chunk)}
    return None


def plot_summary(evaluations, answers, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    colors = {'baseline': '#3468b2', 'sft': '#c16b32'}
    for split, color in [('train', '#3468b2'), ('validation', '#c16b32')]:
        rows = sorted([row for row in evaluations if row['split'] == split], key=lambda row: row['step'])
        axes[0].plot([row['step'] for row in rows], [row['token_weighted_loss'] for row in rows],
                     marker='o', color=color, label=split)
    axes[0].set(title='Teacher-forced target loss', xlabel='Optimizer step', ylabel='Token-weighted loss')
    axes[0].set_xticks([0, 10, 20])
    axes[0].legend(frameon=False)
    indexed = {(row['phase'], row['id']): row for row in answers}
    for number in range(1, 33):
        question_id = f'dev3_{number:02d}'
        lengths = [indexed[phase, question_id]['generated_tokens'] for phase in colors]
        axes[1].plot([0, 1], lengths, color='#a0a0a0', alpha=.35, linewidth=.8)
    for x, phase in enumerate(colors):
        rows = [row for row in answers if row['phase'] == phase]
        values = [row['generated_tokens'] for row in rows]
        axes[1].scatter([x] * len(values), values, s=18, alpha=.55, color=colors[phase])
        axes[1].scatter([x], [statistics.median(values)], marker='_', s=500, color='black', zorder=3)
    axes[1].axhline(768, color='#777777', linestyle='--', linewidth=1)
    axes[1].set(xticks=[0, 1], xticklabels=['baseline', 'SFT'], ylabel='Generated tokens',
                title='32 paired development answers', xlim=(-.35, 1.35))
    metrics = ['Length stops', 'Frozen repeat\nperiod 3–12', 'Post-hoc repeat\nperiod 3–128']
    for phase, shift in [('baseline', -.18), ('sft', .18)]:
        rows = [row for row in answers if row['phase'] == phase]
        counts = [sum(row['finish_reason'] == 'length' for row in rows),
                  sum(row['technical']['has_repetition'] for row in rows),
                  sum(repeated_span(row['answer']) is not None for row in rows)]
        axes[2].bar([i + shift for i in range(3)], counts, width=.34, color=colors[phase], label=phase)
    axes[2].set(xticks=range(3), xticklabels=metrics, ylabel='Answers out of 32', title='Generation diagnostics')
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    figure.savefig(directory / 'learning_and_generation.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--tokenizer-path', help='Local selected-model tokenizer for masks and exact text verification')
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        paths = [name for name in archive.namelist() if name.endswith('/run_metadata.json')]
        require(len(paths) == 1, 'Expected one run metadata file')
        prefix = paths[0].removesuffix('run_metadata.json')

        def read_json(name):
            return json.loads(archive.read(prefix + name))

        def read_rows(name):
            return [json.loads(line) for line in archive.read(prefix + name).decode().splitlines() if line.strip()]

        metadata = read_json('run_metadata.json')
        run_id, commit = metadata['run_id'], metadata['repo_commit']
        require(bool(re.fullmatch(r'qwen3_4b_\d{8}T\d{12}', run_id)), 'Invalid run identifier')
        require(prefix == run_id + '/' and metadata['status'] == 'complete', 'Run is incomplete or root differs')
        require(bool(re.fullmatch(r'[0-9a-f]{40}', commit)), 'Invalid source commit')
        for name, expected in metadata['file_sha256'].items():
            require(digest(archive.read(prefix + 'reproduction/' + name)) == expected, 'Snapshot hash differs: ' + name)
            committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
            require(digest(committed) == expected, 'Snapshot differs from source commit: ' + name)
        settings, selection = metadata['settings'], metadata['selection']
        require(settings == read_json('reproduction/data/sft_v3/pilot_config.json'), 'Runtime settings differ')
        require(selection == read_json('reproduction/' + settings['selection_path']), 'Runtime selection differs')
        require(settings['optimizer_steps'] == 20 and settings['epochs'] == 1, 'Unexpected training schedule')
        require(settings['learning_rate'] == 5e-5 and settings['gradient_accumulation_steps'] == 4, 'Unexpected LR or accumulation')
        require(all(metadata['versions'][name] == version for name, version in settings['packages'].items()), 'Runtime package versions differ')
        require(all(metadata['generation_config'][key] == value for key, value in settings['generation'].items()), 'Runtime generation parameters differ')
        for field, directory in [('corpus_manifest_sha256', settings['corpus_directory']),
                                 ('development_manifest_sha256', settings['development_directory'])]:
            require(digest(archive.read(prefix + 'reproduction/' + directory + '/manifest.json')) == settings[field], 'Data manifest differs')

        executed = json.loads(args.notebook.read_text(encoding='utf-8'))
        original = read_json('reproduction/notebooks/04_sft_v3_pilot.ipynb')
        require(code_asts(executed) == code_asts(original), 'Executed notebook code differs from archived source')
        cells = [cell for cell in executed['cells'] if cell['cell_type'] == 'code']
        require(len(cells) == 9 and all(cell['execution_count'] is not None for cell in cells), 'Notebook is incomplete')
        require(not any(output['output_type'] == 'error' for cell in cells for output in cell.get('outputs', [])), 'Notebook has an exception')
        stdout = '\n'.join(''.join(output.get('text', [])) for cell in cells
                           for output in cell.get('outputs', []) if output['output_type'] == 'stream')
        require(all(value in stdout for value in [run_id, metadata['adapter_sha256'], 'PILOT PIPELINE PASS']), 'Notebook output does not identify this completed run')

        train, validation = read_rows('training_subset.jsonl'), read_rows('validation_subset.jsonl')
        corpus = read_rows('reproduction/' + settings['corpus_directory'] + '/records.jsonl')
        expected_train, expected_validation = select_pilot_records(corpus, selection['system_prompt'])
        require(train == expected_train and validation == expected_validation, 'Runtime subsets differ from the fixed corpus')
        questions = read_rows('reproduction/' + settings['development_directory'] + '/questions.jsonl')
        rubrics = read_rows('reproduction/' + settings['development_directory'] + '/rubric.jsonl')
        sources = read_json('reproduction/' + settings['development_directory'] + '/sources.json')['sources']
        answers = read_rows('answers.jsonl')
        indexed = validate_pilot_records(questions, answers, settings, selection, run_id, complete=True)
        for row in answers:
            require(row['technical'] == technical_indicators(row['question'], row['answer']), 'Technical indicators differ')
            expected_prompt = selection['system_prompt']
            require(row['rendered_prompt'].startswith('<|im_start|>system\n' + expected_prompt), 'Wrong system prompt in generation')
        for question in questions:
            left, right = [indexed[phase, question['id']] for phase in ['baseline', 'sft']]
            for field in ['seed', 'input_token_ids', 'rendered_prompt', 'max_new_tokens', 'eos_token_ids']:
                require(left[field] == right[field], 'Paired generation input differs: ' + field)
        automatic = summarize(questions, {'baseline': selection['system_prompt'], 'sft': selection['system_prompt']}, answers)
        require(automatic == read_json('judge/automatic_summary.json'), 'Automatic summary differs from recomputed metrics')
        require({**automatic, 'quality_status': 'awaiting_judge_scores_and_content_review'} == read_json('automatic_summary.json'), 'Root automatic summary differs')

        steps = read_rows('training_steps.jsonl')
        require([row['step'] for row in steps] == list(range(1, 21)), 'Step sequence differs')
        for row in steps:
            require(len(row['microbatches']) == 4, 'Accumulation group size differs')
            total = sum(item['supervised_tokens'] for item in row['microbatches'])
            require(total == row['supervised_tokens'] and total > 0, 'Training token count differs')
            require(math.isfinite(row['gradient_norm_before_clip']), 'Non-finite gradient norm')
            require(math.isclose(sum(item['loss'] * item['supervised_tokens'] for item in row['microbatches']) / total,
                                 row['token_weighted_loss']), 'Training weighted loss differs')
        visits = Counter(item['sample_id'] for row in steps for item in row['microbatches'])
        require(visits == Counter(row['sample_id'] for row in train), 'Each training example must occur exactly once')
        training_summary = read_json('training_summary.json')
        require(training_summary['optimizer_steps'] == 20 and training_summary['microbatches'] == 80
                and training_summary['epochs_completed'] == 1 and training_summary['each_train_example_used_once']
                and training_summary['nonzero_B_matrices'] == 252, 'Training summary differs')
        evaluations = read_rows('evaluations.jsonl')
        evaluation_index = {(row['step'], row['split']): row for row in evaluations}
        require(len(evaluations) == len(evaluation_index) == 6, 'Missing or duplicate evaluation')
        require(set(evaluation_index) == {(step, split) for step in [0, 10, 20] for split in ['train', 'validation']}, 'Evaluation schedule differs')
        for row in evaluations:
            examples = row['per_example']
            expected_rows = train if row['split'] == 'train' else validation
            require([item['sample_id'] for item in examples] == [item['sample_id'] for item in expected_rows], 'Evaluation IDs differ')
            require(row['n'] == len(examples), 'Evaluation count differs')
            total = sum(item['supervised_tokens'] for item in examples)
            require(total == row['supervised_tokens'], 'Evaluation token count differs')
            require(math.isclose(sum(item['loss'] * item['supervised_tokens'] for item in examples) / total, row['token_weighted_loss']), 'Evaluation loss differs')
            require(math.isclose(sum(item['correct_tokens'] for item in examples) / total, row['token_accuracy']), 'Token accuracy differs')
            require(math.isclose(sum(item['eos_correct'] for item in examples) / len(examples), row['eos_accuracy']), 'EOS accuracy differs')
        fit = read_json('fit_summary.json')
        require(fit['checkpoint_step'] == 20, 'Fit checkpoint differs')
        for split in ['train', 'validation']:
            before, after = [evaluation_index[step, split] for step in [0, 20]]
            expected_fit = {
                'n': before['n'], 'loss_before': before['token_weighted_loss'], 'loss_after': after['token_weighted_loss'],
                'loss_decreased': after['token_weighted_loss'] < before['token_weighted_loss'],
                'token_accuracy_before': before['token_accuracy'], 'token_accuracy_after': after['token_accuracy'],
                'eos_accuracy_before': before['eos_accuracy'], 'eos_accuracy_after': after['eos_accuracy'],
            }
            require(fit['by_split'][split] == expected_fit, 'Fit summary differs')

        weights = archive.read(prefix + 'adapter/adapter_model.safetensors')
        require(digest(weights) == metadata['adapter_sha256'], 'Adapter hash differs')
        tensors = load_safetensors(weights)
        require(len(tensors) == 504 and sum(value.size for value in tensors.values()) == 33030144, 'Saved tensor count or shape differs')
        require(all(np.isfinite(value).all() for value in tensors.values()), 'Non-finite saved weights')
        b_tensors = {name: value for name, value in tensors.items() if '.lora_B.' in name}
        require(len(b_tensors) == 252 and all(np.count_nonzero(value) for value in b_tensors.values()), 'Some B matrices did not change')
        audit = read_json('adapter_audit.json')
        require(audit['adapter_sha256'] == digest(weights) and audit['adapter_bytes'] == len(weights), 'Weight audit hash differs')
        require(audit['saved_tensors'] == len(tensors) and audit['all_finite'], 'Weight audit count differs')
        for item in audit['tensors']:
            value = tensors[item['name']]
            require(list(value.shape) == item['shape'] and item['dtype'] == 'torch.' + str(value.dtype), 'Saved tensor audit differs')
        adapter_config = read_json('adapter/adapter_config.json')
        require(adapter_config['base_model_name_or_path'] == selection['model_id']
                and adapter_config['revision'] == selection['revision'], 'Adapter base/revision differs')
        fresh, influence, roundtrip = [read_json(name) for name in [
            'fresh_adapter_check.json', 'trained_adapter_check.json', 'save_reload_check.json']]
        require(all(fresh[key] for key in ['all_B_zero', 'active_vs_disabled_allclose', 'base_weights_frozen', 'gradient_checkpointing']), 'Fresh adapter controls failed')
        require(influence['disabled_base_allclose'] and influence['disabled_base_max_abs_diff'] == 0
                and influence['trained_adapter_max_abs_diff'] > 0, 'Trained adapter controls failed')
        require(roundtrip['adapter_sha256'] == digest(weights) and roundtrip['all_logits_match']
                and roundtrip['all_greedy_tokens_match'], 'Reload controls failed')
        require([row['question_id'] for row in roundtrip['per_question']] == settings['control_question_ids'], 'Reload control IDs differ')
        require(all(row['logits_allclose'] and row['logits_max_abs_diff'] == 0
                    and row['before_reload_tokens'] == row['after_reload_tokens'] for row in roundtrip['per_question']), 'Reload results differ')

        mapping = read_json('judge/judge_mapping.json')
        requests = read_rows('judge/judge_requests.jsonl')
        require(len(mapping) == len(requests) == 64, 'Judge coverage differs')
        links = {item['request_id']: item for item in mapping}
        require(len(links) == 64 and {(row['prompt_id'], row['question_id']) for row in mapping} == set(indexed), 'Judge mapping differs')
        rubric_index = {row['id']: row for row in rubrics}
        for request in requests:
            link = links[request['request_id']]
            source = indexed[link['prompt_id'], link['question_id']]
            payload = json.loads(request['user_prompt'])
            require(set(payload) == {'question', 'assistant_answer', 'case_criteria', 'source_notes'}, 'Judge payload fields differ')
            require(payload['question'] == source['question'] and payload['assistant_answer'] == source['answer'], 'Judge response text differs')
            require(payload['case_criteria'] == rubric_index[source['id']], 'Judge rubric differs')
            require(request['system_prompt'] == archive.read(prefix + 'reproduction/judge/judge_prompt_v3.md').decode(), 'Judge prompt differs')
            basis_ids = {key for item in payload['case_criteria']['required_criteria'] for key in item['source_ids']}
            require(set(payload['source_notes']) == basis_ids, 'Source note coverage differs')
            require(all(note == {field: sources[key][field] for field in ['title', 'url', 'supports', 'limits']}
                        for key, note in payload['source_notes'].items()), 'Source notes differ')
        require(len({row['request_id'] for row in requests}) == 64, 'Duplicate judge request ID')
        judge_names = ['judge_requests.jsonl', 'judge_system_prompt.md', 'judge_response_schema.json', 'judge_run_config.json']
        with zipfile.ZipFile(io.BytesIO(archive.read(prefix + 'judge/for_judge.zip'))) as judge_zip:
            require(set(judge_zip.namelist()) == set(judge_names), 'Judge ZIP files differ')
            require(all(judge_zip.read(name) == archive.read(prefix + 'judge/' + name) for name in judge_names), 'Judge ZIP content differs')
        checks = read_json('checks.json')
        require(checks['status'] == 'PASS' and checks['saved_answers'] == checks['judge_requests'] == 64
                and checks['optimizer_steps'] == 20 and checks['logged_errors'] == len(read_rows('errors.jsonl')), 'Final checks differ')

        masks = read_json('mask_audit.json')
        controls_decoded = []
        if args.tokenizer_path:
            from transformers import AutoTokenizer
            from checks.sft_smoke_v3 import build_masked_example
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            for split, rows in [('train', train), ('validation', validation)]:
                require([build_masked_example(tokenizer, row, settings['max_seq_length'])[1] for row in rows] == masks[split], 'Recomputed masks differ')
            for row in answers:
                require(tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['answer'], 'Decoded answer differs')
                require(tokenizer.decode(row['generated_token_ids'], skip_special_tokens=False) == row['raw_answer'], 'Raw decoded answer differs')
                inputs = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': selection['system_prompt']},
                    {'role': 'user', 'content': row['question']},
                ], tokenize=True, add_generation_prompt=True, return_dict=False)
                require(inputs == row['input_token_ids'] and len(inputs) == row['input_tokens'], 'Native prompt token IDs differ')
            controls_decoded = [{'question_id': row['question_id'],
                                 'answer': tokenizer.decode(row['before_reload_tokens'], skip_special_tokens=True),
                                 'scope': 'Post-training greedy reload control, max_new_tokens=64; not one of the 64 judge answers'}
                                for row in roundtrip['per_question']]

        directory = ROOT / 'results/sft_v3_pilot' / run_id
        directory.mkdir(parents=True, exist_ok=True)
        filenames = ['mask_audit.json', 'evaluations.jsonl', 'trained_adapter_check.json', 'run_metadata.json',
                     'errors.jsonl', 'training_subset.jsonl', 'training_steps.jsonl', 'validation_subset.jsonl',
                     'answers.jsonl', 'automatic_summary.json', 'fit_summary.json', 'checks.json', 'pip_freeze.txt',
                     'fresh_adapter_check.json', 'save_reload_check.json', 'training_summary.json', 'comparison.md',
                     'adapter_audit.json', 'adapter/adapter_config.json']
        for name in filenames:
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(prefix + name))
        judge_dir = ROOT / 'judge/sft_v3_pilot'
        judge_dir.mkdir(parents=True, exist_ok=True)
        for name in [*judge_names, 'judge_mapping.json', 'review_notes.json', 'for_judge.zip']:
            (judge_dir / name).write_bytes(archive.read(prefix + 'judge/' + name))
        write_json(judge_dir / 'package_manifest.json', {
            'run_id': run_id, 'requests': 64, 'baseline': 32, 'sft': 32, 'status': 'prepared_not_scored',
            'source_archive': args.archive.name, 'source_archive_sha256': sha256_file(args.archive),
            'package_sha256': sha256_file(judge_dir / 'for_judge.zip'),
        })
        summary = {
            'run_id': run_id, 'source_commit': commit, 'fit': {}, 'generation': automatic['by_prompt'],
            'posthoc_repetition_diagnostic': {
                'rule': 'At least three adjacent exact copies of a block of 3..128 words after casefold and word tokenization',
                'added_after_inspecting_outputs': True, 'frozen_autocheck_unchanged': True,
                'by_phase': {phase: [{'question_id': row['id'], 'finish_reason': row['finish_reason'], **match}
                                     for row in answers if row['phase'] == phase and (match := repeated_span(row['answer']))]
                             for phase in ['baseline', 'sft']},
            },
            'paired_changes': {}, 'greedy_reload_controls': controls_decoded,
            'total_wall_seconds': (datetime.fromisoformat(metadata['finished_at_utc']) - datetime.fromisoformat(metadata['started_at_utc'])).total_seconds(),
            'judge_quality_status': 'awaiting_independent_judge_scores',
        }
        for split in ['train', 'validation']:
            before, after = [evaluation_index[step, split] for step in [0, 20]]
            summary['fit'][split] = {
                'loss_before': before['token_weighted_loss'], 'loss_after': after['token_weighted_loss'],
                'relative_loss_drop': 1 - after['token_weighted_loss'] / before['token_weighted_loss'],
                'n_with_lower_loss': sum(a['loss'] < b['loss'] for b, a in zip(before['per_example'], after['per_example'], strict=True)),
                'n': before['n'],
            }
        for metric in ['has_repetition', 'warning_marker_found']:
            gained, lost = [], []
            for question in questions:
                if metric == 'warning_marker_found' and not question['dangerous']:
                    continue
                before, after = [indexed[phase, question['id']]['technical'][metric] for phase in ['baseline', 'sft']]
                if not before and after:
                    gained.append(question['id'])
                if before and not after:
                    lost.append(question['id'])
            summary['paired_changes'][metric] = {'gained': gained, 'lost': lost}
        write_json(directory / 'analysis_summary.json', summary)
        write_json(directory / 'verification.json', {
            'status': 'PASS', 'archive_file': args.archive.name, 'archive_sha256': sha256_file(args.archive),
            'notebook_file': args.notebook.name, 'notebook_sha256': sha256_file(args.notebook),
            'matching_run_id': run_id, 'matching_source_commit': commit,
            'source_snapshot_hashes_verified_against_commit': len(metadata['file_sha256']),
            'executed_code_matches_snapshot': True, 'executed_code_cells': len(cells),
            'adapter_sha256': digest(weights), 'saved_tensors': len(tensors),
            'saved_parameters': sum(value.size for value in tensors.values()), 'saved_weights_finite': True,
            'nonzero_B_matrices': len(b_tensors), 'saved_tensor_dtypes': dict(Counter(str(value.dtype) for value in tensors.values())),
            'reload_tokens_verified_on_n_questions': 3, 'masks_recomputed': bool(args.tokenizer_path),
            'answer_text_and_prompt_tokens_recomputed': bool(args.tokenizer_path),
            'evaluation_and_generation_aggregates_recomputed': True, 'training_sample_visits': dict(visits),
            'judge_request_texts_and_nested_zip_verified': 64,
            'limits': 'Recorded numerical GPU controls checked for consistency; no fresh full-model inference performed by this analyzer.',
        })
        plot_summary(evaluations, answers, directory.parent)
        print(json.dumps({'verification': 'PASS', 'run_id': run_id, 'fit': summary['fit'],
                          'posthoc_repetition_counts': {key: len(value) for key, value in summary['posthoc_repetition_diagnostic']['by_phase'].items()},
                          'output': str(directory), 'judge_package': str(judge_dir / 'for_judge.zip')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
