"""Verify adapter-strength diagnostics and export the 16 newly generated judge cases."""

import argparse
import ast
import hashlib
import json
import re
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import export_artifacts, finish_reason, read_jsonl, sha256_file, write_json
from script.analyze_sft_v3_pilot import repeated_span

STRENGTHS = [0.0, 0.25, 0.5, 1.0]
IDS = ['dev3_01', 'dev3_07', 'dev3_15', 'dev3_21', 'dev3_24', 'dev3_25', 'dev3_26', 'dev3_29']
CONTROLS = ['dev3_01', 'dev3_07', 'dev3_25']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_asts(notebook):
    return [ast.dump(ast.parse(''.join(cell['source'])), include_attributes=False)
            for cell in notebook['cells'] if cell['cell_type'] == 'code']


def numbered_repeat(answer):
    """Additional post-hoc diagnostic; remove Markdown list counters before matching."""
    text = re.sub(r'(?m)^\s*\d+[.)]\s+', '', answer)
    return repeated_span(text)


def plot_lengths(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    colors = ['#3468b2', '#46936a', '#b18a35', '#bd6258']
    width = .19
    indexed = {(row['adapter_multiplier'], row['question_id']): row for row in rows}
    for i, (strength, color) in enumerate(zip(STRENGTHS, colors)):
        axes[0].bar([j + (i - 1.5) * width for j in range(8)],
                    [indexed[strength, qid]['generated_tokens'] for qid in IDS], width=width,
                    label=str(strength), color=color)
    axes[0].axhline(768, color='#777777', linestyle='--', linewidth=1)
    axes[0].set(xticks=range(8), xticklabels=[q.removeprefix('dev3_') for q in IDS],
                xlabel='Diagnostic question ID', ylabel='Generated tokens', title='32 matched sampling answers')
    for i, strength in enumerate(STRENGTHS):
        group = [row for row in rows if row['adapter_multiplier'] == strength]
        counts = [sum(row['finish_reason'] == 'length' for row in group),
                  sum(repeated_span(row['answer']) is not None for row in group),
                  sum(numbered_repeat(row['answer']) is not None for row in group)]
        axes[1].bar([j + (i - 1.5) * width for j in range(3)], counts, width=width, color=colors[i])
    axes[1].set(xticks=range(3), xticklabels=['Length stop', 'Exact repeat\n3–128 words', 'Repeat after\nlist-counter removal'],
                ylabel='Answers out of 8', title='Generation diagnostics', ylim=(0, 3))
    axes[1].set_yticks([0, 1, 2, 3])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, title='LoRA multiplier', ncols=4,
                   frameon=False, fontsize=8, title_fontsize=9, loc='upper center')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    figure.savefig(directory / 'strength_comparison.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--tokenizer-path', help='Optional local tokenizer for text and native prompt verification')
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        candidates = [name for name in archive.namelist() if name.endswith('/run_metadata.json')]
        require(len(candidates) == 1, 'Expected one run metadata file')
        prefix = candidates[0].removesuffix('run_metadata.json')

        def read_json(name):
            return json.loads(archive.read(prefix + name))

        def read_rows(name):
            return [json.loads(line) for line in archive.read(prefix + name).decode().splitlines() if line.strip()]

        metadata = read_json('run_metadata.json')
        run_id = metadata['run_id']
        require(re.fullmatch(r'adapter_strength_v1_\d{8}T\d{12}', run_id), 'Unexpected run ID')
        require(prefix == run_id + '/' and metadata['status'] == 'complete_awaiting_content_review', 'Run is incomplete')
        require(metadata['strengths'] == STRENGTHS and metadata['diagnostic_ids'] == IDS
                and metadata['greedy_ids'] == CONTROLS, 'Diagnostic conditions differ')
        for name, expected in metadata['source_hashes'].items():
            source = archive.read(prefix + 'source/' + name)
            require(hashlib.sha256(source).hexdigest() == expected, 'Source snapshot hash differs: ' + name)
            committed = subprocess.check_output(['git', 'show', f"{metadata['source_commit']}:{name}"], cwd=ROOT)
            require(source == committed, 'Source differs from pinned commit: ' + name)
        require(metadata['selection'] == read_json('source/data/sft_v3/selection.json')
                and metadata['generation'] == read_json('source/data/sft_v3/pilot_config.json')['generation'], 'Runtime configuration differs')
        prep = json.loads((ROOT / 'results/adapter_strength_v1/preparation.json').read_text())
        notebook_path = ROOT / 'notebooks/05_adapter_strength_diagnostic_v1.ipynb'
        require(sha256_file(notebook_path) == prep['notebook_sha256'], 'Published diagnostic source changed')
        executed = json.loads(args.notebook.read_text())
        require(code_asts(executed) == code_asts(json.loads(notebook_path.read_text())), 'Executed code differs from prepared notebook')
        cells = [cell for cell in executed['cells'] if cell['cell_type'] == 'code']
        require(len(cells) == 7 and all(cell.get('execution_count') is not None for cell in cells), 'Notebook is incomplete')
        require(not any(output['output_type'] == 'error' for cell in cells for output in cell.get('outputs', [])), 'Notebook has an exception')
        stdout = '\n'.join(''.join(o.get('text', [])) for cell in cells for o in cell.get('outputs', []) if o['output_type'] == 'stream')
        require(run_id in stdout and 'TECHNICAL PASS' in stdout, 'Notebook does not identify the completed run')
        require(metadata['adapter_sha256'] == '461961f38e7552aa6d4c6990b5c1aaffea5801d24b1a4d4a4b86fda61feb98f8', 'Wrong adapter')
        require(read_json('adapter_config.json') == read_json('source/results/sft_v3_pilot/qwen3_4b_20261118T113449101907/adapter/adapter_config.json'), 'Adapter configuration differs')
        require(read_json('parameter_hashes_before.json') == read_json('parameter_hashes_after.json'), 'Recorded parameter hashes changed')
        parameter_count = len(read_json('parameter_hashes_before.json'))
        require(parameter_count > 0, 'Empty parameter hash snapshot')
        scale_api = read_json('peft_scale_api.json')
        require(scale_api['n_layers'] == 252 and all(v == 2 for v in scale_api['initial_scaling'].values()), 'Initial scales differ')
        require(scale_api['peft_version'] == metadata['versions']['peft'] == '0.21.2', 'PEFT version differs')
        verification = read_json('verification.json')
        require(verification['status'] == 'TECHNICAL_PASS_NOT_QUALITY_CLAIM' and verification['weights_unchanged'], 'Technical controls failed')
        require(set(verification['zero_vs_disabled_logits']) == set(IDS), 'Zero-control coverage differs')
        require(all(v['max_abs_diff'] == 0 and v['allclose'] for v in verification['zero_vs_disabled_logits'].values()), 'Zero differs from disabled base')
        require(all(v == 0 for v in verification['absolute_scale_cycle_max_abs_diff'].values()), 'Scale restoration differs')
        require(all(verification['one_restored_logits'].values()), 'Scale-one logits were not restored')
        require(set(verification['absolute_scale_cycle_max_abs_diff']) == set(verification['one_restored_logits']) == set(IDS), 'Logit control coverage differs')

        questions = read_rows('source/data/development_v3/questions.jsonl')
        selected_questions = [q for q in questions if q['id'] in IDS]
        indexed_questions = {q['id']: q for q in questions}
        seeds = {q['id']: 20261116 + i for i, q in enumerate(questions, 1)}
        baseline = read_rows('source/results/sft_v3_pilot/qwen3_4b_20261118T113449101907/answers.jsonl')
        baseline_index = {(row['phase'], row['id']): row for row in baseline}
        answers = read_rows('answers.jsonl')
        controls = read_rows('greedy_controls.jsonl')
        reference_zero = {row['question_id']: row for row in read_rows('disabled_reference_controls.jsonl')}
        reference_one = {row['question_id']: row for row in read_rows('scale1_reference_controls.jsonl')}
        audit = read_json('input_audit.json')
        for rows, ids, kind, limit in [(answers, IDS, 'sampling', 768), (controls, CONTROLS, 'greedy', 64)]:
            expected = {(strength, qid) for strength in STRENGTHS for qid in ids}
            require(len(rows) == len(expected) and {(row['adapter_multiplier'], row['question_id']) for row in rows} == expected, 'Missing or duplicate generated records')
            for row in rows:
                qid = row['question_id']
                require(row['seed'] == seeds[qid] == audit[qid]['seed'], 'Question seed differs')
                require(row['question'] == indexed_questions[qid]['question'] and row['category'] == indexed_questions[qid]['category']
                        and row['dangerous'] == indexed_questions[qid]['dangerous'], 'Question metadata differs')
                require(row['input_token_ids'] == audit[qid]['input_token_ids'] == baseline_index['baseline', qid]['input_token_ids'], 'Prompt tokens differ')
                require(row['generation_kind'] == kind and row['max_new_tokens'] == limit, 'Generation settings differ')
                require(row['generated_tokens'] == len(row['generated_token_ids']), 'Generated token count differs')
                require(row['finish_reason'] == finish_reason(row['generated_token_ids'], [151645, 151643], limit), 'Finish reason differs')
        for qid in CONTROLS:
            zero = next(row for row in controls if row['adapter_multiplier'] == 0 and row['question_id'] == qid)
            one = next(row for row in controls if row['adapter_multiplier'] == 1 and row['question_id'] == qid)
            require(zero['generated_token_ids'] == reference_zero[qid]['generated_token_ids'], 'Zero greedy differs from disabled reference')
            require(one['generated_token_ids'] == reference_one[qid]['generated_token_ids'], 'Restored greedy differs from initial scale one')
        endpoint_matches = {}
        for strength, phase in [(0.0, 'baseline'), (1.0, 'sft')]:
            endpoint_matches[str(strength)] = [row['question_id'] for row in answers
                if row['adapter_multiplier'] == strength and row['generated_token_ids'] == baseline_index[phase, row['question_id']]['generated_token_ids']]
            require(len(endpoint_matches[str(strength)]) == 8, 'Historical endpoint tokens differ')

        if args.tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
            for row in answers + controls + list(reference_zero.values()) + list(reference_one.values()):
                require(tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['answer'], 'Decoded answer differs')
                require(tokenizer.decode(row['generated_token_ids'], skip_special_tokens=False) == row['raw_answer'], 'Raw decoded answer differs')
            for qid in IDS:
                tokens = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': metadata['selection']['system_prompt']},
                    {'role': 'user', 'content': indexed_questions[qid]['question']},
                ], tokenize=True, add_generation_prompt=True, return_dict=False)
                require(tokens == audit[qid]['input_token_ids'], 'Native chat template differs')

        summary = {'run_id': run_id, 'by_scale': [], 'historical_endpoint_token_matches': endpoint_matches,
                   'total_wall_seconds': (datetime.fromisoformat(metadata['completed_at_utc']) - datetime.fromisoformat(metadata['started_at_utc'])).total_seconds(),
                   'numbered_repeat_diagnostic': 'Post-hoc: remove line-leading Markdown list counters, then seek three adjacent exact copies with period 3..128 words.',
                   'independent_middle_scale_judge_status': 'not_scored'}
        for strength in STRENGTHS:
            group = [row for row in answers if row['adapter_multiplier'] == strength]
            summary['by_scale'].append({
                'multiplier': strength, 'n': len(group), 'length_stops': sum(row['finish_reason'] == 'length' for row in group),
                'repetition_ids': [row['question_id'] for row in group if repeated_span(row['answer'])],
                'repetition_after_list_counter_removal_ids': [row['question_id'] for row in group if numbered_repeat(row['answer'])],
                'median_tokens': statistics.median(row['generated_tokens'] for row in group),
                'median_words': statistics.median(len(re.findall(r'\w+', row['answer'])) for row in group),
                'generation_seconds': sum(row['elapsed_seconds'] for row in group),
            })
        recorded_summary = read_json('technical_summary.json')
        for actual, saved in zip(summary['by_scale'], recorded_summary['by_scale']):
            require(actual['length_stops'] == saved['length_stops'] and actual['repetition_ids'] == saved['repetition_question_ids'], 'Recorded generation summary differs')
        directory = ROOT / 'results/adapter_strength_v1' / run_id
        directory.mkdir(parents=True, exist_ok=True)
        for name in archive.namelist():
            relative = name.removeprefix(prefix)
            if not name.endswith('/') and not relative.startswith('source/'):
                require('/' not in relative and relative != name, 'Unexpected output path')
                (directory / relative).write_bytes(archive.read(name))
        write_json(directory / 'analysis_summary.json', summary)
        write_json(directory / 'artifact_verification.json', {
            'status': 'PASS', 'archive_sha256': sha256_file(args.archive), 'notebook_sha256': sha256_file(args.notebook),
            'published_notebook_sha256': sha256_file(notebook_path), 'executed_code_matches_published': True,
            'executed_code_cells': len(cells), 'source_hashes_verified_against_commit': len(metadata['source_hashes']),
            'matching_parameter_fingerprint_entries': parameter_count, 'main_answers': len(answers), 'greedy_controls': len(controls),
            'historical_endpoint_tokens_match': True, 'text_and_prompt_tokenizer_checks': bool(args.tokenizer_path),
            'scope': 'Offline archive consistency and recorded control verification; no new full-model inference.'})
        plot_lengths(answers, directory.parent)

        judge_dir = ROOT / 'judge/adapter_strength_v1'
        judge_dir.mkdir(parents=True, exist_ok=True)
        new_rows = [{**indexed_questions[row['question_id']], **row, 'id': row['question_id'],
                     'prompt_id': 'scale_' + str(row['adapter_multiplier']), 'eos_token_ids': [151645, 151643]}
                    for row in answers if row['adapter_multiplier'] in [0.25, 0.5]]
        rubric = [row for row in read_rows('source/data/development_v3/rubric.jsonl') if row['id'] in IDS]
        sources = read_json('source/data/development_v3/sources.json')['sources']
        schema = json.loads((ROOT / 'judge/judge_response_schema_v3.json').read_text())
        count = export_artifacts(judge_dir, selected_questions,
            {'scale_0.25': metadata['selection']['system_prompt'], 'scale_0.5': metadata['selection']['system_prompt']},
            new_rows, rubric, sources, (ROOT / 'judge/judge_prompt_v3.md').read_text(), schema, run_id, 20261118)
        require(count == 16, 'Expected sixteen new answers for the judge')
        write_json(judge_dir / 'judge_run_config.json', {
            'status': 'prepared_not_scored', 'model': 'Qwen3.8-27B', 'temperature': 0, 'max_tokens': 2048,
            'expected_requests': count, 'system_prompt_path': 'judge_system_prompt.md',
            'response_schema_path': 'judge_response_schema.json',
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'household_judge_v3', 'strict': True, 'schema': schema}}})
        with zipfile.ZipFile(judge_dir / 'for_judge.zip', 'w', zipfile.ZIP_DEFLATED) as zipped:
            for name in ['judge_requests.jsonl', 'judge_system_prompt.md', 'judge_response_schema.json', 'judge_run_config.json']:
                zipped.write(judge_dir / name, arcname=name)
        old_mapping = {(row['prompt_id'], row['question_id']): row
                       for row in json.loads((ROOT / 'judge/sft_v3_pilot/judge_mapping.json').read_text())}
        prior_scores = {key: {row['request_id']: row for row in read_jsonl(ROOT / 'results/sft_v3_pilot/judges' / (key + '_scores.jsonl'))}
                        for key in ['qwen', 'gemma']}
        reused = []
        for strength, phase in [(0.0, 'baseline'), (1.0, 'sft')]:
            for qid in IDS:
                uid = old_mapping[phase, qid]['request_id']
                reused.append({'multiplier': strength, 'question_id': qid, 'prior_request_id': uid,
                               'prior_candidate_run': baseline_index[phase, qid]['run_id'],
                               'generated_token_ids_identical': True,
                               'judges': {key: {'scores_path': 'results/sft_v3_pilot/judges/' + key + '_scores.jsonl',
                                                'semantic_state': prior_scores[key][uid]['semantic_state']}
                                          for key in prior_scores}})
        write_json(judge_dir / 'endpoint_score_reuse.json', reused)
        write_json(judge_dir / 'package_manifest.json', {
            'status': 'prepared_not_scored', 'run_id': run_id, 'new_requests': 16,
            'new_multipliers': [0.25, 0.5], 'endpoint_answers_identical_to_previously_scored': len(reused),
            'source_archive_sha256': sha256_file(args.archive), 'judge_zip_sha256': sha256_file(judge_dir / 'for_judge.zip')})
        print(json.dumps({'verification': 'PASS', **summary, 'new_judge_requests': count}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
