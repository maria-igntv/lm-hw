"""Verify a completed notebook-06 run and reproduce its technical comparison offline."""

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.adapter_expanded_v1 import (
    build_expanded_history, expanded_repeated_span, export_expanded_artifacts,
    export_expanded_reuse, validate_expanded_records,
)
from checks.screening_v3 import read_jsonl, sha256_file, summarize, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def asts(notebook):
    return [ast.dump(ast.parse(''.join(c['source'])), include_attributes=False)
            for c in notebook['cells'] if c['cell_type'] == 'code']


def plot(summary, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = summary['technical_by_condition']
    names = ['Baseline', 'Full SFT', 'SFT × 0.25']
    keys = ['baseline', 'sft', 'scale_0.25']
    colors = ['#3468b2', '#ba6258', '#46936a']
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout='constrained')
    fields = [('mean_generated_tokens', 'Mean generated tokens'), ('length_stops', 'Length stops / 32'),
              ('numbered_repeat_count', 'Repeats after list-counter removal / 32')]
    for ax, (field, title) in zip(axes, fields):
        values = [groups[k][field] for k in keys]
        ax.bar(names, values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=9)
        for i,v in enumerate(values): ax.text(i, v, f'{v:.1f}' if field.startswith('mean') else str(v), ha='center', va='bottom')
        ax.set_ylim(0, max(values) * 1.2 + 1)
    fig.savefig(directory / 'technical_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--judge-archive', type=Path)
    parser.add_argument('--pilot-archive', type=Path)
    parser.add_argument('--tokenizer-path', type=Path)
    args = parser.parse_args()
    preparation = json.loads((ROOT / 'results/adapter_expanded_v1/preparation.json').read_text())
    original_nb = ROOT / 'notebooks/06_adapter_expanded_v1.ipynb'
    require(sha256_file(original_nb) == preparation['notebook_sha256'], 'Prepared notebook changed')
    executed = json.loads(args.notebook.read_text())
    require(asts(executed) == asts(json.loads(original_nb.read_text())), 'Executed code differs from prepared notebook')
    code_cells = [c for c in executed['cells'] if c['cell_type'] == 'code']
    require(len(code_cells) == 8 and all(c.get('execution_count') is not None for c in code_cells), 'Notebook is incomplete')
    require(not any(o['output_type'] == 'error' for c in code_cells for o in c.get('outputs', [])), 'Notebook has an exception')
    stdout = '\n'.join(''.join(o.get('text', [])) for c in code_cells for o in c.get('outputs', []) if o['output_type'] == 'stream')
    with tempfile.TemporaryDirectory(prefix='lm_hw_expanded_verify_') as temporary, zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        meta_names = [n for n in names if n.endswith('/run_metadata.json') and '/source/' not in n]
        require(len(meta_names) == 1, 'Expected one run metadata')
        prefix = meta_names[0].removesuffix('run_metadata.json')
        directory = Path(temporary) / prefix
        for name in names:
            parts = PurePosixPath(name).parts
            require(name.startswith(prefix) and not name.startswith('/') and '..' not in parts and '\\' not in name,
                    'Unsafe ZIP member')
            if name.endswith('/'): continue
            info = archive.getinfo(name)
            require(info.file_size < 20_000_000, 'Unexpected oversized non-weight artifact')
            target = Path(temporary) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        source = directory / 'source'
        metadata = json.loads((directory / 'run_metadata.json').read_text())
        run_id = metadata['run_id']
        require(re.fullmatch(r'adapter_expanded_v1_\d{8}T\d{12}', run_id) and prefix == run_id + '/', 'Unexpected run ID')
        require(metadata['status'] == 'complete_awaiting_judge' and metadata['total_answers'] == 64
                and metadata['new_answers_saved'] == 24 and metadata['reused_answers'] == 40, 'Incomplete run')
        require(run_id in stdout and 'CONTROLS PASS' in stdout and '24 /24' in stdout, 'Notebook stdout does not identify completed run')
        require(metadata['source_commit'] == preparation['source_commit']
                and metadata['source_hashes'] == preparation['source_hashes']
                and metadata['embedded_helper_sha256'] == preparation['helper_sha256'], 'Pinned inputs differ')
        for relative, expected in metadata['source_hashes'].items():
            data = (source / relative).read_bytes()
            require(hashlib.sha256(data).hexdigest() == expected, 'Source hash differs: ' + relative)
            committed = subprocess.check_output(['git', 'show', f"{metadata['source_commit']}:{relative}"], cwd=ROOT)
            require(data == committed, 'Source differs from commit: ' + relative)
        questions = read_jsonl(source / 'data/development_v3/questions.jsonl')
        settings = json.loads((source / 'data/sft_v3/pilot_config.json').read_text())
        selection = json.loads((source / 'data/sft_v3/selection.json').read_text())
        choice = json.loads((source / 'data/sft_v3/adapter_strength_selection.json').read_text())
        require(metadata['selection'] == selection and metadata['candidate_selection'] == choice
                and metadata['adapter_multiplier'] == 0.25 and metadata['generation'] == settings['generation'], 'Runtime settings differ')
        prior_root = source / 'results/sft_v3_pilot/qwen3_4b_20261118T113449101907'
        diagnostic_root = source / 'results/adapter_strength_v1/adapter_strength_v1_20261118T200443561241'
        prior_rows = read_jsonl(prior_root / 'answers.jsonl')
        history = build_expanded_history(questions, settings, selection, prior_rows,
            read_jsonl(diagnostic_root / 'answers.jsonl'), run_id)
        rows = read_jsonl(directory / 'answers.jsonl')
        validate_expanded_records(questions, settings, selection, run_id, rows, history, complete=True)
        input_audit = json.loads((directory / 'input_audit.json').read_text())
        baseline = {r['id']:r for r in history if r['prompt_id'] == 'baseline'}
        require(set(input_audit) == set(baseline), 'Input audit coverage differs')
        for row in rows:
            expected = baseline[row['id']]
            require(row['input_token_ids'] == input_audit[row['id']]['input_token_ids'] == expected['input_token_ids']
                    and row['seed'] == input_audit[row['id']]['seed'] == expected['seed'], 'Input tokens or seed differ')
        verification = json.loads((directory / 'verification.json').read_text())
        controls = read_jsonl(directory / ('controls_' + verification['control_session'] + '.jsonl'))
        keys = {(r['prompt_id'], r['id']) for r in controls}
        require(len(controls) == len(keys) == 6 and keys == {(p,q) for p in ['baseline','scale_0.25'] for q in metadata['control_ids']}, 'Control coverage differs')
        historical = {(r['prompt_id'],r['id']):r for r in history}
        for row in controls:
            old = historical[row['prompt_id'],row['id']]
            require(all(row[f] == old[f] for f in ['answer','raw_answer','generated_token_ids','input_token_ids','seed']), 'Control text/tokens differ')
        require(verification['status'] == 'TECHNICAL_PASS_NOT_QUALITY_CLAIM'
                and all(verification['token_matches'].values()) and len(verification['token_matches']) == 6
                and all(verification['zero_vs_disabled'].values()) and all(verification['scale_restoration'].values())
                and verification['lora_parameters_unchanged'] and verification['adapter_file_unchanged'], 'Recorded technical controls failed')
        before = json.loads((directory / 'lora_hashes_before.json').read_text())
        require(before == json.loads((directory / 'lora_hashes_after.json').read_text()) and len(before) == 504, 'LoRA hashes differ')
        scaling = json.loads((directory / 'scaling_audit.json').read_text())
        require(scaling['n_layers'] == len(scaling['original_scales']) == 252
                and set(scaling['original_scales'].values()) == {2.0}
                and scaling['target_multiplier'] == .25 and scaling['target_effective_scale'] == .5, 'Scaling audit differs')
        inference = json.loads((directory / 'inference_config.json').read_text())
        require(inference['adapter_multiplier'] == .25 and inference['generation'] == settings['generation']
                and inference['system_prompt'] == selection['system_prompt']
                and metadata['adapter_sha256'] == inference['adapter_sha256'] == choice['adapter_sha256'], 'Inference configuration differs')
        if args.pilot_archive:
            require(sha256_file(args.pilot_archive) == metadata['input_pilot_zip_sha256'], 'Original pilot ZIP hash differs')
            with zipfile.ZipFile(args.pilot_archive) as pilot_zip:
                with pilot_zip.open('qwen3_4b_20261118T113449101907/adapter/adapter_model.safetensors') as weights:
                    h = hashlib.sha256()
                    for block in iter(lambda: weights.read(8 << 20), b''): h.update(block)
                require(h.hexdigest() == choice['adapter_sha256'], 'Original adapter weights SHA differs')
        if args.tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
            for row in rows + controls:
                require(tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['answer']
                        and tokenizer.decode(row['generated_token_ids'], skip_special_tokens=False) == row['raw_answer'], 'Decoded text differs')
        expected_summary = json.loads((directory / 'automatic_summary.json').read_text())
        old_requests = read_jsonl(directory / 'judge/judge_requests.jsonl')
        old_mapping = json.loads((directory / 'judge/judge_mapping.json').read_text())
        old_reused = read_jsonl(directory / 'reused_judge_scores.jsonl')
        rubrics = read_jsonl(source / 'data/development_v3/rubric.jsonl')
        sources = json.loads((source / 'data/development_v3/sources.json').read_text())['sources']
        regenerated = Path(temporary) / 'recomputed'; regenerated.mkdir()
        count = export_expanded_artifacts(regenerated, questions, rows, history, settings, selection, run_id,
            rubrics, sources, (source / 'judge/judge_prompt_v3.md').read_text(),
            json.loads((source / 'judge/judge_response_schema_v3.json').read_text()))
        require(count == 24 and read_jsonl(regenerated / 'judge/judge_requests.jsonl') == old_requests
                and json.loads((regenerated / 'judge/judge_mapping.json').read_text()) == old_mapping,
                'Recomputed judge requests differ')
        require(json.loads((regenerated / 'automatic_summary.json').read_text()) == expected_summary, 'Recomputed automatic summary differs')
        require(export_expanded_reuse(regenerated, source, history, rubrics, sources) == old_reused, 'Reused scores differ')
        provided_judge = args.judge_archive or directory / 'judge/for_judge.zip'
        with zipfile.ZipFile(provided_judge) as judge_zip:
            require(set(judge_zip.namelist()) == {'judge_requests.jsonl','judge_system_prompt.md','judge_response_schema.json','judge_run_config.json'}, 'Judge ZIP files differ')
            for name in judge_zip.namelist():
                require(judge_zip.read(name) == (regenerated / 'judge' / name).read_bytes(), 'Judge ZIP content differs: ' + name)
        technical = dict(expected_summary['by_prompt'])
        full = summarize(questions, {'sft':selection['system_prompt']}, [r for r in prior_rows if r['phase']=='sft'])['by_prompt']['sft']
        full['repeat_3_to_128_words_ids'] = [r['id'] for r in prior_rows if r['phase']=='sft' and expanded_repeated_span(r['answer'])]
        technical['sft'] = full
        for phase,group in technical.items():
            condition_rows = [r for r in (prior_rows if phase=='sft' else rows) if r['prompt_id']==phase]
            repeated = [r['id'] for r in condition_rows if expanded_repeated_span(re.sub(r'(?m)^\s*\d+[.)]\s+', '', r['answer']))]
            group['numbered_repeat_count'] = len(repeated)
            group['numbered_repeat_ids'] = repeated
        summary = {'run_id':run_id, 'status':'technical_complete_judge_pending', 'n_questions':32,
            'new_answers':24, 'reused_answers':40, 'sampling_controls':6,
            'technical_by_condition':technical,
            'new_generation_minutes':sum(r['elapsed_seconds'] for r in rows if r['origin']=='new')/60,
            'run_wall_minutes':(datetime.fromisoformat(metadata['completed_at_utc'])-datetime.fromisoformat(metadata['started_at_utc'])).total_seconds()/60,
            'candidate_token_reduction_fraction':1-technical['scale_0.25']['mean_generated_tokens']/technical['baseline']['mean_generated_tokens'],
            'new_judge_scores_received':0, 'new_judge_scores_expected_per_judge':24,
            'quality_conclusion':'More concise and fewer length stops; overall safety/helpfulness improvement not established.'}
        receipt = {'status':'PASS', 'archive_sha256':sha256_file(args.archive),
            'executed_notebook_sha256':sha256_file(args.notebook), 'published_notebook_sha256':sha256_file(original_nb),
            'executed_code_matches_published':True, 'executed_code_cells':8, 'notebook_exceptions':0,
            'source_files_verified_against_commit':len(metadata['source_hashes']),
            'main_answers':64, 'historical_answers_unchanged':40, 'control_output_tokens_matched':6,
            'matching_lora_hash_entries':len(before), 'external_judge_zip_verified':bool(args.judge_archive),
            'original_adapter_weights_independently_hashed':bool(args.pilot_archive),
            'local_tokenizer_decode_checks':bool(args.tokenizer_path),
            'new_judge_requests_verified':24, 'reused_judge_scores_verified':80,
            'analyzer_sha256':sha256_file(Path(__file__)),
            'scope':'Offline verification of files and recorded GPU controls; no new model/API calls.'}
        out = ROOT / 'results/adapter_expanded_v1' / run_id
        out.mkdir(parents=True, exist_ok=True)
        for path in directory.rglob('*'):
            if path.is_file() and 'source' not in path.relative_to(directory).parts:
                target = out / path.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path,target)
        write_json(out / 'artifact_verification.json', receipt)
        write_json(out / 'analysis_summary.json', summary)
        # Keep plain execution logs, without widget/HTML/JavaScript state.
        (out / 'execution_stdout.txt').write_text(re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', stdout), encoding='utf-8')
        plot(summary, out.parent)
        print(json.dumps({'verification':'PASS', 'run_id':run_id, 'answers':64,
            'controls_matched':6, 'judge_requests':24, 'new_judge_scores':0,
            'summary':str(out / 'analysis_summary.json')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
