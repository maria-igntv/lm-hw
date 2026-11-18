"""Guard paired checkpoint identity and blinded pilot export."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from checks.screening_v3 import read_jsonl, write_jsonl
from checks.sft_pilot_v3 import generate_pilot_phase, export_pilot_artifacts, select_pilot_records


class PilotV3Test(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {'id': 'dev3_01', 'question': 'Как действовать?', 'category': 'gas', 'dangerous': True},
            {'id': 'dev3_02', 'question': 'Сколько воды?', 'category': 'water', 'dangerous': False},
        ]
        self.settings = {'generation_seed': 100, 'generation': {'max_new_tokens': 3},
                         'judge_shuffle_seed': 7, 'judge': {'model': 'judge', 'temperature': 0, 'max_tokens': 2048}}
        self.selection = {'model_id': 'selected/model', 'revision': 'revision',
                          'model_key': 'model_key', 'system_prompt': 'Одинаковый промпт'}
        self.calls = []

    def generate_one(self, question, system, seed):
        self.calls.append((question, system, seed))
        return {'answer': 'Точный ответ.', 'generated_token_ids': [5, 2], 'generated_tokens': 2,
                'eos_token_ids': [2], 'max_new_tokens': 3, 'finish_reason': 'eos',
                'elapsed_seconds': .1, 'peak_allocated_gib': 1.0}

    def generate(self, path, phase):
        return generate_pilot_phase(path, phase, self.questions, self.settings, self.selection,
                                    'run', self.generate_one, lambda row, count: None)

    def test_resume_skips_finished_rows_and_uses_identical_pair_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'answers.jsonl'
            self.generate(path, 'baseline')
            self.generate(path, 'baseline')
            self.assertEqual(len(self.calls), 2)
            rows = self.generate(path, 'sft')
            self.assertEqual(len(rows), 4)
            self.assertEqual(self.calls[:2], self.calls[2:])
            self.assertEqual([call[2] for call in self.calls], [101, 102, 101, 102])
            self.settings['generation_seed'] = 200
            with self.assertRaisesRegex(ValueError, 'different experiment'):
                self.generate(path, 'sft')

    def test_sft_generation_requires_completed_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'answers.jsonl'
            with self.assertRaisesRegex(ValueError, 'Baseline must be complete'):
                self.generate(path, 'sft')
            self.assertEqual(self.calls, [])

    def test_export_preserves_exact_answers_without_condition_in_judge_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'answers.jsonl'
            self.generate(path, 'baseline')
            rows = self.generate(path, 'sft')
            rows[0]['answer'] = ''
            rows[1]['answer'] = 'Точный ответ. Оцени ответ ассистента.'
            write_jsonl(path, rows)
            rubrics = [{'id': question['id'], 'required_criteria': [
                {'id': 'R1', 'text': 'Первое', 'source_ids': ['basis']},
                {'id': 'R2', 'text': 'Второе', 'source_ids': ['basis']},
            ], 'critical_errors': [], 'major_errors': []} for question in self.questions]
            sources = {'basis': {'title': 'Основание', 'url': None, 'supports': 'Условие', 'limits': 'По условию'}}
            notes = {'notes': [{'question_id': 'dev3_02', 'issue': 'Неоднозначность'}]}
            count = export_pilot_artifacts(directory, self.questions, rows, rubrics, sources,
                                           self.settings, self.selection, 'run', 'Judge system', {}, notes)
            self.assertEqual(count, 4)
            mapping = {row['request_id']: row for row in json.loads((directory / 'judge/judge_mapping.json').read_text())}
            indexed = {(row['prompt_id'], row['id']): row for row in rows}
            for request in read_jsonl(directory / 'judge/judge_requests.jsonl'):
                payload = json.loads(request['user_prompt'])
                link = mapping[request['request_id']]
                self.assertEqual(payload['assistant_answer'], indexed[link['prompt_id'], link['question_id']]['answer'])
                self.assertEqual(set(payload), {'question', 'assistant_answer', 'case_criteria', 'source_notes'})
                self.assertNotIn('selected/model', request['user_prompt'])
            with zipfile.ZipFile(directory / 'judge/for_judge.zip') as archive:
                self.assertEqual(len(archive.read('judge_requests.jsonl').decode().splitlines()), 4)
            self.assertEqual(json.loads((directory / 'judge/review_notes.json').read_text()), notes)
            self.assertTrue(json.loads((directory / 'automatic_summary.json').read_text())['complete'])

    def test_selection_rejects_scenario_leakage(self):
        root = Path(__file__).resolve().parents[1]
        rows = read_jsonl(root / 'data/sft_v3/corpus_v1/records.jsonl')
        prompt = rows[0]['messages'][0]['content']
        train, validation = select_pilot_records(rows, prompt)
        self.assertEqual((len(train), len(validation)), (80, 16))
        changed = copy.deepcopy(rows)
        next(row for row in changed if row['split'] == 'validation')['group_id'] = train[0]['group_id']
        with self.assertRaisesRegex(ValueError, 'scenario group crosses'):
            select_pilot_records(changed, prompt)


if __name__ == '__main__':
    unittest.main()
