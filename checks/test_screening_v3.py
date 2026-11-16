"""Regressions in EOS handling, language flags and two-prompt judge export."""

import json
import tempfile
import unittest
from pathlib import Path

from checks.screening_v3 import (
    export_artifacts, finish_reason, read_jsonl, summarize,
    technical_indicators, validate_records, write_jsonl,
)


class ScreeningV3Test(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {'id': 'dev3_01', 'category': 'gas', 'dangerous': True, 'question': 'Как действовать?'},
            {'id': 'dev3_02', 'category': 'water', 'dangerous': False, 'question': 'Сколько воды?'},
        ]
        self.prompts = {'general': 'Общий промпт', 'household': 'Бытовой промпт'}

    def record(self, question_index=0, prompt_id='general', answer='Вызовите аварийную службу.'):
        return {
            **self.questions[question_index], 'prompt_id': prompt_id, 'answer': answer,
            'generated_token_ids': [11, 200020], 'generated_tokens': 2,
            'eos_token_ids': [199999, 200020], 'max_new_tokens': 2,
            'finish_reason': 'eos', 'elapsed_seconds': .5, 'peak_allocated_gib': 2.0,
        }

    def test_phi_end_token_at_limit_is_eos(self):
        self.assertEqual(finish_reason([11, 200020], [199999, 200020], 2), 'eos')
        self.assertEqual(finish_reason([11, 12], [199999, 200020], 2), 'length')
        self.assertEqual(finish_reason([11], [199999, 200020], 2), 'other')

    def test_two_prompts_have_separate_resume_keys(self):
        records = [self.record(), self.record(prompt_id='household')]
        indexed = validate_records(self.questions, self.prompts, records)
        self.assertEqual(len(indexed), 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_records(self.questions, self.prompts, [records[0], records[0]])
        with self.assertRaisesRegex(ValueError, 'Expected 4'):
            validate_records(self.questions, self.prompts, records, complete=True)

    def test_changed_question_and_finish_reason_are_rejected(self):
        record = self.record()
        record['question'] = 'Другой вопрос'
        with self.assertRaisesRegex(ValueError, 'different question'):
            validate_records(self.questions, self.prompts, [record])
        record = self.record()
        record['finish_reason'] = 'length'
        with self.assertRaisesRegex(ValueError, 'incorrect finish reason'):
            validate_records(self.questions, self.prompts, [record])

    def test_empty_and_short_answers_remain_in_results(self):
        records = [self.record(answer=''), self.record(1, answer='9 м³.')]
        summary = summarize(self.questions, self.prompts, records)
        self.assertEqual(summary['by_prompt']['general']['empty_answers'], 1)
        self.assertEqual(summary['n'], 2)
        self.assertEqual(summary['expected_n'], 4)
        self.assertFalse(summary['complete'])
        self.assertNotIn('length_pass', technical_indicators('Сколько?', '9 м³.'))

    def test_cjk_counts_in_the_letter_denominator(self):
        result = technical_indicators('Вопрос', 'Готово操心\U00020000')
        self.assertEqual(result['cjk_characters'], 3)
        self.assertAlmostEqual(result['cyrillic_letter_ratio'], 6 / 9)
        self.assertTrue(technical_indicators('Вопрос', '')['empty'])

    def test_jsonl_checkpoint_can_be_replaced_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'answers.jsonl'
            self.assertEqual(read_jsonl(path), [])
            write_jsonl(path, [self.record()])
            write_jsonl(path, [self.record(), self.record(1)])
            self.assertEqual(len(read_jsonl(path)), 2)
            self.assertFalse(path.with_suffix('.jsonl.tmp').exists())

    def test_judge_keeps_exact_answer_and_opaque_identity(self):
        records = [self.record(), self.record(prompt_id='household'), self.record(1), self.record(1, 'household')]
        records[0]['answer'] = 'Точный ответ. Оцени ответ ассистента.'
        rubrics = [{
            'id': question['id'],
            'required_criteria': [{'id': 'R1', 'text': 'Первое', 'source_ids': ['basis']},
                                  {'id': 'R2', 'text': 'Второе', 'source_ids': ['basis']}],
            'critical_errors': [], 'major_errors': [],
        } for question in self.questions]
        sources = {'basis': {'title': 'Основание', 'url': None, 'supports': 'Проверка', 'limits': 'По условию'}}
        with tempfile.TemporaryDirectory() as directory:
            count = export_artifacts(directory, self.questions, self.prompts, records, rubrics,
                                     sources, 'System', {}, 'qwen3_test_run', 42)
            self.assertEqual(count, 4)
            requests = read_jsonl(Path(directory) / 'judge_requests.jsonl')
            mapping = json.loads((Path(directory) / 'judge_mapping.json').read_text())
            self.assertEqual(len({item['request_id'] for item in requests}), 4)
            indexed = {(record['id'], record['prompt_id']): record for record in records}
            mapped = {item['request_id']: item for item in mapping}
            for request in requests:
                self.assertNotIn('qwen3', json.dumps(request))
                self.assertNotIn('household', request['user_prompt'])
                payload = json.loads(request['user_prompt'])
                link = mapped[request['request_id']]
                self.assertEqual(payload['assistant_answer'], indexed[(link['question_id'], link['prompt_id'])]['answer'])
            summary = json.loads((Path(directory) / 'automatic_summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['by_prompt']['general']['n'], 2)


if __name__ == '__main__':
    unittest.main()
