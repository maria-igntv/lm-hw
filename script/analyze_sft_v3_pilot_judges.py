"""Verify/import a continuation package or recompute published pilot judge metrics."""

import argparse
import hashlib
import io
import json
import random
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.screening_v3 import read_jsonl, sha256_file, write_json, write_jsonl
from script.analyze_judges_v3 import condition_summary, semantic_state

COMMIT = '89dd84aaf40feac11563802b100a97076fb9b72e'
RUN = 'qwen3_4b_20261118T113449101907'
DEFAULT_OUTPUT = ROOT / 'results/sft_v3_pilot/judges'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(rows, questions, excluded):
    index = {(row['phase'], row['question_id']): row for row in rows}
    pairs = [(index['baseline', q['id']], index['sft', q['id']])
             for q in questions if q['id'] not in excluded]
    result = {'n': len(pairs), 'semantic_transitions': dict(Counter(
        before['semantic_state'] + '->' + after['semantic_state'] for before, after in pairs))}
    for field in ['critical_errors', 'major_errors', 'unsupported_specifics']:
        result[field] = {
            'fixed_ids': [a['question_id'] for b, a in pairs if b['score'][field] and not a['score'][field]],
            'new_ids': [a['question_id'] for b, a in pairs if not b['score'][field] and a['score'][field]],
        }
    for metric in ['helpfulness', 'safety']:
        deltas = [a['score'][metric] - b['score'][metric] for b, a in pairs]
        rng = random.Random(20261118)
        bootstrap_means = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
        result[metric] = {'mean_delta': mean(deltas), 'improved': sum(d > 0 for d in deltas),
                          'unchanged': sum(d == 0 for d in deltas), 'worsened': sum(d < 0 for d in deltas),
                          'paired_question_bootstrap_95pct': [bootstrap_means[249], bootstrap_means[9749]]}
    return result


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = summary['all']
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), layout='constrained')
    labels, before, after = [], [], []
    for judge, name in [('qwen', 'Qwen'), ('gemma', 'Gemma')]:
        for metric in ['helpfulness', 'safety']:
            labels.append(name + '\n' + metric)
            before.append(values[judge]['by_phase']['baseline']['means'][metric])
            after.append(values[judge]['by_phase']['sft']['means'][metric])
    for rows, shift, color, name in [(before, -.18, '#3468b2', 'baseline'), (after, .18, '#c16b32', 'SFT')]:
        bars = axes[0].bar([i + shift for i in range(4)], rows, .34, color=color, label=name)
        axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    axes[0].set(xticks=range(4), xticklabels=labels, ylim=(0, 5.25),
                ylabel='Judge score (1–5)', title='Mean helpfulness and safety')
    axes[0].legend(frameon=False, fontsize=9)
    conditions = [values[judge]['by_phase'][phase] for judge in ['qwen', 'gemma'] for phase in ['baseline', 'sft']]
    bottom = [0] * 4
    for field, name, color in [('pass', 'Pass', '#4b9569'), ('review', 'Review', '#a8aeb6'), ('fail', 'Fail', '#bf6259')]:
        counts = [row[field] for row in conditions]
        bars = axes[1].bar(range(4), counts, bottom=bottom, color=color, label=name)
        axes[1].bar_label(bars, labels=[str(n) if n else '' for n in counts], label_type='center', color='white', fontsize=10)
        bottom = [a + b for a, b in zip(bottom, counts)]
    axes[1].set(xticks=range(4), xticklabels=['Qwen\nbaseline', 'Qwen\nSFT', 'Gemma\nbaseline', 'Gemma\nSFT'],
                ylim=(0, 35.5), ylabel='Answers out of 32', title='Frozen semantic rule: pass / review / fail')
    axes[1].legend(frameon=False, fontsize=9, ncols=3, loc='upper center')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    figure.savefig(output / 'judge_comparison.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional original lm_hw_continuation_package.zip')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    judge_dir = ROOT / 'judge/sft_v3_pilot'
    requests = {row['request_id']: row for row in read_jsonl(judge_dir / 'judge_requests.jsonl')}
    mapping = {row['request_id']: row for row in json.loads((judge_dir / 'judge_mapping.json').read_text())}
    questions = read_jsonl(ROOT / 'data/development_v3/questions.jsonl')
    question_index = {row['id']: row for row in questions}
    schema = json.loads((judge_dir / 'judge_response_schema.json').read_text())
    validator = Draft202012Validator(schema)
    score_rows, raw_rows, public_runs = {}, {}, []
    verification = None

    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            prefixes = [name.removesuffix('checksums.json') for name in archive.namelist()
                        if name.endswith('/checksums.json')]
            require(len(prefixes) == 1, 'Expected one checksums manifest')
            prefix = prefixes[0]

            def read_json(name):
                return json.loads(archive.read(prefix + name))

            def rows(name):
                return [json.loads(line) for line in archive.read(prefix + name).decode().splitlines() if line.strip()]

            hashes = read_json('checksums.json')
            for name, digest in hashes.items():
                require(hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest, 'Package hash differs: ' + name)
            source_files = [name for name in archive.namelist() if name.startswith(prefix + 'stage_a/source/') and not name.endswith('/')]
            for name in source_files:
                relative = name.removeprefix(prefix + 'stage_a/source/')
                committed = subprocess.check_output(['git', 'show', f'{COMMIT}:{relative}'], cwd=ROOT)
                require(archive.read(name) == committed, 'Source differs from pinned commit: ' + relative)
            original_package = archive.read(prefix + 'stage_a/source/judge/sft_v3_pilot/for_judge.zip')
            require(hashlib.sha256(original_package).hexdigest() == sha256_file(judge_dir / 'for_judge.zip'), 'Submitted judge package differs')
            with zipfile.ZipFile(io.BytesIO(original_package)) as nested:
                require(all(nested.read(name) == (judge_dir / name).read_bytes() for name in nested.namelist()), 'Nested judge payload differs')
            models = {row['name']: row for row in read_json('stage_a/raw/model_metadata.json')}
            for key, model_name in [('qwen', 'Qwen3.8-27B'), ('gemma', 'gemma-4-31B-it')]:
                status = read_json(f'stage_a/raw/{key}_64_status.json')
                configuration = status['configuration']
                require(status['status'] == 'success' and status['model_name'] == model_name, 'Run status/model differs')
                require(configuration['inference_params']['temperature'] == 0
                        and configuration['inference_params']['max_tokens'] == 2048, 'Judge inference parameters differ')
                require(configuration['response_format']['json_schema']['schema'] == schema, 'Effective judge schema differs')
                require(configuration['system_prompt'] == next(iter(requests.values()))['system_prompt'], 'Effective judge prompt differs')
                require(not rows(f'stage_a/raw/{key}_64_failed.jsonl'), 'Judge pipeline has failed rows')
                raw_rows[key] = rows(f'stage_a/raw/{key}_64_output.jsonl')
                source_scores = rows(f'stage_a/{key}_results.jsonl')
                require(all(row['run_id'] == status['id'] for row in source_scores), 'Score run identity differs')
                score_rows[key] = [{k: v for k, v in row.items() if k not in {'run_id', 'reasoning'}}
                                   | {'service_batch': key + '_64'} for row in source_scores]
                catalog = models[model_name]
                public_runs.append({
                    'service_batch': key + '_64', 'model': model_name, 'status': status['status'],
                    'count': len(raw_rows[key]), 'failed_count': 0,
                    'created_at': status['created_at'], 'updated_at': status['updated_at'],
                    'parameters': configuration['inference_params'], 'response_format': configuration['response_format'],
                    'catalog': {field: catalog[field] for field in ['name', 'provider', 'engine', 'backend', 'aliases'] if field in catalog},
                    'usage': {field: status.get(field) for field in ['total_tokens', 'prompt_tokens', 'completion_tokens']},
                    'worker_build_version': 'not reported by the service',
                })
            verification = {'status': 'PASS', 'source_commit': COMMIT,
                            'input_archive': args.archive.name, 'input_archive_sha256': sha256_file(args.archive),
                            'package_files_with_verified_hashes': len(hashes),
                            'source_files_verified_against_commit': len(source_files),
                            'judge_package_sha256': hashlib.sha256(original_package).hexdigest(),
                            'score_values_manually_modified': False,
                            'scope': 'Offline consistency verification of supplied inputs, raw responses and recorded service metadata; no new model calls.'}
            (output / 'analysis.html').write_bytes(archive.read(prefix + 'stage_a/analysis.html'))
            (output / 'source_interpretation.txt').write_bytes(archive.read(prefix + 'stage_a/interpretation.txt'))
    else:
        score_rows = {key: read_jsonl(output / (key + '_scores.jsonl')) for key in ['qwen', 'gemma']}
        raw_rows = {key: read_jsonl(output / 'raw' / (key + '_output.jsonl')) for key in ['qwen', 'gemma']}
        public_runs = json.loads((output / 'runs.json').read_text())

    normalized = {}
    for key in ['qwen', 'gemma']:
        raw = {row['uid']: row for row in raw_rows[key]}
        require(len(raw) == len(raw_rows[key]) == 64, 'Missing or duplicate raw judge results')
        scores = score_rows[key]
        require(len(scores) == len({row['request_id'] for row in scores}) == 64, 'Missing or duplicate parsed results')
        require(set(raw) == {row['request_id'] for row in scores} == set(requests), 'Judge coverage differs from submitted requests')
        for row in scores:
            uid = row['request_id']
            link = mapping[uid]
            require(row['candidate'] == link and row['phase'] == link['prompt_id']
                    and row['question_id'] == link['question_id'], 'Candidate mapping differs')
            require(row['dangerous'] == question_index[row['question_id']]['dangerous']
                    and row['category'] == question_index[row['question_id']]['category'], 'Candidate metadata differs')
            validator.validate(row['score'])
            require(json.loads(raw[uid]['model_response']) == row['score']
                    and raw[uid]['model_response'] == row['raw_response'], 'Parsed/raw score differs')
            expected = Counter(c['id'] for c in json.loads(requests[uid]['user_prompt'])['case_criteria']['required_criteria'])
            require(Counter(c['criterion_id'] for c in row['score']['criteria_checks']) == expected, 'Required criterion coverage differs')
            require(not row['score']['critical_errors'] or row['score']['safety'] == 1, 'Critical/safety consistency differs')
            require(row['semantic_state'] == semantic_state(row['score']), 'Semantic state differs from frozen rule')
        normalized[key] = scores

    def summarize(excluded=()):
        result = {}
        for key, rows in normalized.items():
            allowed = [q for q in questions if q['id'] not in excluded]
            selected = [row for row in rows if row['question_id'] not in excluded]
            result[key] = {
                'by_phase': {phase: condition_summary([row for row in selected if row['phase'] == phase], len(allowed))
                             for phase in ['baseline', 'sft']},
                'by_risk': {risk: {phase: condition_summary(
                    [row for row in selected if row['phase'] == phase and row['dangerous'] == danger],
                    sum(q['dangerous'] == danger for q in allowed)) for phase in ['baseline', 'sft']}
                    for risk, danger in [('dangerous', True), ('ordinary', False)]},
                'paired': paired(rows, questions, excluded),
            }
        return result

    summary = {'protocol': 'judge_v3.0', 'source_candidate_run': RUN, 'all': summarize(),
               'excluding_dev3_02': summarize({'dev3_02'}),
               'excluding_dev3_24_posthoc': summarize({'dev3_24'}),
               'interpretation': 'Scores and severity labels are judge outputs. Exclusion of dev3_24 is a post-hoc sensitivity check, not a revision of raw scores.',
               'bootstrap': {'seed': 20261118, 'replicates': 10000,
                             'scope': 'Question resampling within selected development; not seed variability or a fresh test.'}}
    disagreements = []
    left, right = [{row['request_id']: row for row in normalized[key]} for key in ['qwen', 'gemma']]
    for uid in sorted(left):
        a, b = left[uid], right[uid]
        if a['semantic_state'] != b['semantic_state'] or bool(a['score']['critical_errors']) != bool(b['score']['critical_errors']):
            disagreements.append({'request_id': uid, 'question_id': a['question_id'], 'phase': a['phase'],
                                  'qwen_state': a['semantic_state'], 'gemma_state': b['semantic_state'],
                                  'qwen_critical': a['score']['critical_errors'], 'gemma_critical': b['score']['critical_errors']})
    summary['between_judge_disagreements'] = disagreements
    (output / 'raw').mkdir(exist_ok=True)
    for key in normalized:
        write_jsonl(output / (key + '_scores.jsonl'), normalized[key])
        write_jsonl(output / 'raw' / (key + '_output.jsonl'), raw_rows[key])
    write_json(output / 'runs.json', public_runs)
    write_json(output / 'summary.json', summary)
    write_jsonl(output / 'review_queue.jsonl', [row for rows in normalized.values() for row in rows if row['review_flags']])
    write_jsonl(output / 'errors.jsonl', [])
    if verification:
        verification['valid_results'] = {key: len(rows) for key, rows in normalized.items()}
        verification['json_schema_and_criterion_coverage_verified'] = True
        verification['parsed_and_raw_responses_identical'] = True
        verification['aggregate_rule'] = 'Existing semantic_state and condition_summary from script/analyze_judges_v3.py'
        write_json(output / 'verification.json', verification)
    plot_summary(summary, output)
    print(json.dumps({key: item['by_phase'] for key, item in summary['all'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
