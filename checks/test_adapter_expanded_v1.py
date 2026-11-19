"""Check reuse, interruption recovery and exact blinded judge payloads offline."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from checks.adapter_expanded_v1 import (
    EXPANDED_PHASE, build_expanded_history, export_expanded_artifacts,
    export_expanded_reuse, generate_expanded_remaining, validate_expanded_records,
)
from checks.screening_v3 import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


class ExpandedAdapterTest(unittest.TestCase):
    def setUp(self):
        self.questions = read_jsonl(ROOT / 'data/development_v3/questions.jsonl')
        self.settings = json.loads((ROOT / 'data/sft_v3/pilot_config.json').read_text())
        self.selection = json.loads((ROOT / 'data/sft_v3/selection.json').read_text())
        self.run = 'expanded_test'
        self.history = build_expanded_history(self.questions, self.settings, self.selection,
            read_jsonl(ROOT / 'results/sft_v3_pilot/qwen3_4b_20261118T113449101907/answers.jsonl'),
            read_jsonl(ROOT / 'results/adapter_strength_v1/adapter_strength_v1_20261118T200443561241/answers.jsonl'), self.run)
        self.calls = []

    def generate_one(self, question, system, seed):
        self.calls.append(question['id'])
        return {'answer': 'Точный неизменённый ответ.', 'generated_token_ids': [10, 151645],
                'generated_tokens': 2, 'eos_token_ids': [151645, 151643],
                'max_new_tokens': 768, 'finish_reason': 'eos', 'input_token_ids': [1, 2],
                'elapsed_seconds': 0.1, 'peak_allocated_gib': 1.0}

    def generate(self, directory, on_saved=lambda row, n: None):
        return generate_expanded_remaining(Path(directory) / 'answers.jsonl', self.questions,
            self.settings, self.selection, self.run, self.history, self.generate_one, on_saved)

    def test_interrupted_run_resumes_without_regenerating_saved_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            def interrupt(row, n):
                if n == 2:
                    raise InterruptedError('Simulated runtime interruption')
            with self.assertRaises(InterruptedError):
                self.generate(directory, interrupt)
            self.assertEqual(len(read_jsonl(Path(directory) / 'answers.jsonl')), 42)
            rows = self.generate(directory)
            self.generate(directory)
            self.assertEqual(len(rows), 64)
            self.assertEqual(len(self.calls), 24)
            self.assertEqual(len(set(self.calls)), 24)
            self.assertEqual(sum(r['origin'] == 'reused' for r in rows), 40)

    def test_modified_history_or_settings_cannot_be_silently_reused(self):
        changed = copy.deepcopy(self.history)
        changed[0]['answer'] += ' Изменено.'
        with self.assertRaisesRegex(ValueError, 'Historical answer'):
            validate_expanded_records(self.questions, self.settings, self.selection,
                                      self.run, changed, self.history)
        settings = copy.deepcopy(self.settings)
        settings['generation_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'different experiment'):
            validate_expanded_records(self.questions, settings, self.selection,
                                      self.run, self.history, self.history)

    def test_export_contains_only_24_new_exact_answers_and_no_condition_in_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.generate(directory)
            rubrics = read_jsonl(ROOT / 'data/development_v3/rubric.jsonl')
            sources = json.loads((ROOT / 'data/development_v3/sources.json').read_text())['sources']
            count = export_expanded_artifacts(directory, self.questions, rows, self.history,
                self.settings, self.selection, self.run, rubrics, sources,
                (ROOT / 'judge/judge_prompt_v3.md').read_text(),
                json.loads((ROOT / 'judge/judge_response_schema_v3.json').read_text()))
            self.assertEqual(count, 24)
            judge_dir = Path(directory) / 'judge'
            mapping = {r['request_id']: r for r in json.loads((judge_dir / 'judge_mapping.json').read_text())}
            by_id = {r['id']: r for r in rows if r['origin'] == 'new'}
            for request in read_jsonl(judge_dir / 'judge_requests.jsonl'):
                payload = json.loads(request['user_prompt'])
                self.assertEqual(set(payload), {'question', 'assistant_answer', 'case_criteria', 'source_notes'})
                qid = mapping[request['request_id']]['question_id']
                self.assertEqual(payload['assistant_answer'], by_id[qid]['answer'])
                self.assertNotIn(EXPANDED_PHASE, request['user_prompt'])
            with zipfile.ZipFile(judge_dir / 'for_judge.zip') as archive:
                self.assertEqual(len(archive.namelist()), 4)
                self.assertNotIn('judge_mapping.json', archive.namelist())

    def test_old_judgements_match_frozen_text_criteria_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = export_expanded_reuse(directory, ROOT, self.history,
                read_jsonl(ROOT / 'data/development_v3/rubric.jsonl'),
                json.loads((ROOT / 'data/development_v3/sources.json').read_text())['sources'])
            self.assertEqual(len(scores), 80)
            self.assertEqual(sum(r['judge'] == 'qwen' for r in scores), 40)
            changed = copy.deepcopy(self.history)
            changed[0]['answer'] += ' Изменено.'
            with self.assertRaisesRegex(ValueError, 'judged payload differs'):
                export_expanded_reuse(directory, ROOT, changed,
                    read_jsonl(ROOT / 'data/development_v3/rubric.jsonl'),
                    json.loads((ROOT / 'data/development_v3/sources.json').read_text())['sources'])


if __name__ == '__main__':
    unittest.main()
