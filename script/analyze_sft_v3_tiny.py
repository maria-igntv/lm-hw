"""Verify a completed tiny-training archive and summarize measured fit."""

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load as load_safetensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import finish_reason, sha256_file, technical_indicators, write_json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def code_asts(notebook):
    return [ast.dump(ast.parse(''.join(cell['source'])), include_attributes=False)
            for cell in notebook['cells'] if cell['cell_type'] == 'code']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plot_summary(evaluations, phases, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    for split, color, linestyle in [('train', '#3468b2', '-'), ('validation', '#d17a23', '--')]:
        rows = sorted([row for row in evaluations if row['split'] == split], key=lambda row: row['step'])
        axes[0].plot([row['step'] for row in rows], [row['token_weighted_loss'] for row in rows],
                     marker='o', color=color, linestyle=linestyle, label=split)
    axes[0].set(title='Teacher-forced target loss', xlabel='Optimizer step', ylabel='Token-weighted loss')
    axes[0].set_xticks([0, 4, 12, 24])
    axes[0].legend(frameon=False)
    x = list(range(8))
    for phase, shift, color in [('before', -.18, '#3468b2'), ('after', .18, '#3f9566')]:
        axes[1].bar([value + shift for value in x], [row['generated_tokens'] for row in phases[phase]],
                    width=.34, color=color, label=phase)
    axes[1].set_xticks(x, [row['sample_id'].removeprefix('sft3_') for row in phases['before']])
    axes[1].set(title='Greedy answer length on trained questions', xlabel='Sample ID', ylabel='Generated tokens')
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    figure.savefig(directory / 'learning_curves.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--tokenizer-path', help='Optional local selected-model tokenizer to rebuild masks')
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        metadata_paths = [name for name in archive.namelist() if name.endswith('/run_metadata.json')]
        require(len(metadata_paths) == 1, 'Expected one run metadata file')
        prefix = metadata_paths[0].removesuffix('run_metadata.json')
        metadata = json.loads(archive.read(prefix + 'run_metadata.json'))
        run_id = metadata['run_id']
        require(bool(re.fullmatch(r'qwen3_4b_\d{8}T\d{12}', run_id)), 'Unexpected run identifier')
        require(prefix == run_id + '/', 'Archive root differs from run identifier')
        require(metadata['status'] == 'complete', 'Run is incomplete')
        commit = metadata['repo_commit']
        require(bool(re.fullmatch(r'[0-9a-f]{40}', commit)), 'Invalid source commit')
        for filename, expected_hash in metadata['file_sha256'].items():
            snapshot = archive.read(prefix + 'reproduction/' + filename)
            require(digest(snapshot) == expected_hash, 'Snapshot hash differs for ' + filename)
            committed = subprocess.check_output(['git', 'show', f'{commit}:{filename}'], cwd=ROOT)
            require(digest(committed) == expected_hash, 'Snapshot differs from source commit for ' + filename)

        executed = json.loads(args.notebook.read_text(encoding='utf-8'))
        original = json.loads(archive.read(prefix + 'reproduction/notebooks/03_sft_v3_tiny.ipynb'))
        require(code_asts(executed) == code_asts(original), 'Executed code differs from snapshot')
        cells = [cell for cell in executed['cells'] if cell['cell_type'] == 'code']
        require(len(cells) == 9 and all(cell['execution_count'] is not None for cell in cells), 'Notebook is incomplete')
        require(not any(output['output_type'] == 'error' for cell in cells for output in cell.get('outputs', [])), 'Notebook contains an exception')
        stdout = '\n'.join(''.join(output.get('text', [])) for cell in cells
                           for output in cell.get('outputs', []) if output['output_type'] == 'stream')
        require(run_id in stdout and metadata['adapter_sha256'] in stdout and 'TINY PIPELINE PASS' in stdout,
                'Notebook output does not identify the completed archive run')

        weights = archive.read(prefix + 'adapter/adapter_model.safetensors')
        require(digest(weights) == metadata['adapter_sha256'], 'Adapter SHA-256 differs')
        tensors = load_safetensors(weights)
        require(len(tensors) == 504 and sum(value.size for value in tensors.values()) == 33030144, 'Adapter shape differs')
        require(all(np.isfinite(value).all() for value in tensors.values()), 'Saved weights contain NaN or Inf')
        b_tensors = {key: value for key, value in tensors.items() if '.lora_B.' in key}
        require(len(b_tensors) == 252 and all(np.count_nonzero(value) for value in b_tensors.values()), 'B matrices are incomplete or zero')

        def read_rows(filename):
            return [json.loads(line) for line in archive.read(prefix + filename).decode().splitlines()]

        train = read_rows('training_subset.jsonl')
        validation = read_rows('validation_subset.jsonl')
        settings = metadata['settings']
        require([row['sample_id'] for row in train] == settings['sample_ids'], 'Training selection differs')
        require(len(train) == 8 and len(validation) == 16, 'Subset counts differ')
        require(all(row['split'] == 'train' for row in train), 'A training record belongs to another split')
        require(all(row['split'] == 'validation' for row in validation), 'Validation split differs')
        require(not {row['group_id'] for row in train} & {row['group_id'] for row in validation}, 'A scenario crosses split')
        corpus_snapshot = archive.read(prefix + 'reproduction/' + settings['corpus_directory'] + '/records.jsonl')
        original_records = {row['sample_id']: row for row in map(json.loads, corpus_snapshot.decode().splitlines())}
        require(all(row == original_records[row['sample_id']] for row in train + validation), 'Runtime examples differ from frozen corpus')
        index = {row['sample_id']: row for row in train}
        phases = {phase: read_rows('answers_' + phase + '.jsonl') for phase in ['before', 'after', 'after_reload']}
        for phase, rows in phases.items():
            require([row['sample_id'] for row in rows] == settings['sample_ids'], 'Answer coverage differs')
            for row in rows:
                require(row['phase'] == phase and row['question'] == index[row['sample_id']]['messages'][1]['content'], 'Answer identity differs')
                require(row['generated_tokens'] == len(row['generated_token_ids']), 'Generated token count differs')
                require(row['finish_reason'] == finish_reason(row['generated_token_ids'], row['eos_token_ids'], row['max_new_tokens']), 'Finish reason differs')
                require(row['technical_indicators'] == technical_indicators(row['question'], row['answer']), 'Technical indicators differ')
        require(all(left['generated_token_ids'] == right['generated_token_ids'] for left, right in zip(phases['after'], phases['after_reload'])),
                'Generated sequences differ after reload')
        steps = read_rows('training_steps.jsonl')
        require([row['step'] for row in steps] == list(range(1, 25)), 'Training step sequence differs')
        for row in steps:
            total = sum(item['supervised_tokens'] for item in row['microbatches'])
            require(total == row['supervised_tokens'], 'Training token count differs')
            require(math.isfinite(row['gradient_norm_before_clip']), 'Gradient norm is not finite')
            require(math.isclose(sum(item['loss'] * item['supervised_tokens'] for item in row['microbatches']) / total,
                                 row['token_weighted_loss']), 'Training weighted loss differs')
        visits = Counter(item['sample_id'] for row in steps for item in row['microbatches'])
        require(visits == Counter({sample_id: 12 for sample_id in settings['sample_ids']}), 'Training sample frequencies differ')
        evaluations = read_rows('evaluations.jsonl')
        evaluation_index = {(row['step'], row['split']): row for row in evaluations}
        require(len(evaluations) == len(evaluation_index) == 6, 'Duplicate or missing evaluations')
        require(set(evaluation_index) == {(0, 'train'), (0, 'validation'), (4, 'train'), (12, 'train'), (24, 'train'), (24, 'validation')},
                'Evaluation checkpoints differ')
        for row in evaluations:
            examples = row['per_example']
            expected_ids = [item['sample_id'] for item in (train if row['split'] == 'train' else validation)]
            require([item['sample_id'] for item in examples] == expected_ids and row['n'] == len(examples), 'Evaluation identity differs')
            total = sum(item['supervised_tokens'] for item in examples)
            require(total == row['supervised_tokens'], 'Evaluation token count differs')
            require(math.isclose(sum(item['loss'] * item['supervised_tokens'] for item in examples) / total, row['token_weighted_loss']), 'Weighted loss differs')
            require(math.isclose(sum(item['correct_tokens'] for item in examples) / total, row['token_accuracy']), 'Token accuracy differs')
            require(math.isclose(sum(item['eos_correct'] for item in examples) / len(examples), row['eos_accuracy']), 'EOS accuracy differs')
        masks = json.loads(archive.read(prefix + 'mask_audit.json'))
        checks = json.loads(archive.read(prefix + 'checks.json'))
        require(checks['status'] == 'PASS' and checks['optimizer_steps'] == 24, 'Technical checks did not pass')
        require(checks['logged_errors'] == len(read_rows('errors.jsonl')), 'Error count differs')
        if args.tokenizer_path:
            from transformers import AutoTokenizer
            from checks.sft_smoke_v3 import build_masked_example
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            for split, rows in [('train', train), ('validation', validation)]:
                actual = [build_masked_example(tokenizer, row, settings['max_seq_length'])[1] for row in rows]
                require(actual == masks[split], 'Recomputed masks differ')

        directory = ROOT / 'results/sft_v3_tiny' / run_id
        directory.mkdir(parents=True, exist_ok=True)
        filenames = ['mask_audit.json', 'evaluations.jsonl', 'trained_adapter_check.json', 'run_metadata.json',
                     'errors.jsonl', 'training_subset.jsonl', 'training_steps.jsonl', 'validation_subset.jsonl',
                     'answers_before.jsonl', 'answers_after.jsonl', 'answers_after_reload.jsonl', 'fit_summary.json',
                     'checks.json', 'pip_freeze.txt', 'fresh_adapter_check.json', 'save_reload_check.json',
                     'training_summary.json', 'comparison.md', 'adapter/adapter_config.json']
        for filename in filenames:
            destination = directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(prefix + filename))
        initial_train, final_train = evaluation_index[0, 'train'], evaluation_index[24, 'train']
        initial_val, final_val = evaluation_index[0, 'validation'], evaluation_index[24, 'validation']
        summary = {
            'run_id': run_id, 'source_commit': commit,
            'train_loss_relative_drop': 1 - final_train['token_weighted_loss'] / initial_train['token_weighted_loss'],
            'validation_loss_relative_drop': 1 - final_val['token_weighted_loss'] / initial_val['token_weighted_loss'],
            'validation_examples_with_lower_loss': sum(after['loss'] < before['loss'] for before, after in zip(initial_val['per_example'], final_val['per_example'])),
            'answer_statistics': {phase: {
                'n': len(rows), 'median_words': statistics.median(row['technical_indicators']['word_count'] for row in rows),
                'median_tokens': statistics.median(row['generated_tokens'] for row in rows),
                'length_stops': sum(row['finish_reason'] == 'length' for row in rows),
            } for phase, rows in phases.items()},
            'train_evaluations': [row for row in evaluations if row['split'] == 'train'],
            'validation_evaluations': [row for row in evaluations if row['split'] == 'validation'],
            'interpretation': 'Teacher-forced fit on eight trained and sixteen held-out scenarios. Free generation is available for eight trained questions.',
        }
        write_json(directory / 'analysis_summary.json', summary)
        write_json(directory / 'verification.json', {
            'status': 'PASS', 'archive_file': args.archive.name, 'archive_sha256': sha256_file(args.archive),
            'notebook_file': args.notebook.name, 'notebook_sha256': sha256_file(args.notebook),
            'matching_run_id': run_id, 'matching_source_commit': commit,
            'source_snapshot_hashes_verified_against_commit': len(metadata['file_sha256']),
            'executed_code_matches_snapshot': True, 'executed_code_cells': len(cells),
            'adapter_sha256': digest(weights), 'saved_tensors': len(tensors), 'saved_parameters': sum(value.size for value in tensors.values()),
            'saved_weights_finite': True, 'nonzero_B_matrices': len(b_tensors),
            'saved_tensor_dtypes': dict(Counter(str(value.dtype) for value in tensors.values())),
            'reload_tokens_verified_on_n_questions': 8, 'masks_recomputed': bool(args.tokenizer_path),
            'evaluation_aggregates_recomputed': True, 'training_sample_visits': dict(visits),
        })
        plot_summary(evaluations, phases, directory.parent)
        print(json.dumps({'verification': 'PASS', 'run_id': run_id,
                          'train_loss_relative_drop': summary['train_loss_relative_drop'],
                          'validation_loss_relative_drop': summary['validation_loss_relative_drop'],
                          'output': str(directory)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
