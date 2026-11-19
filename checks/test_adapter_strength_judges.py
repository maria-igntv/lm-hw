"""Keep the real retry trace auditable without making model/API calls."""

import shutil
import tempfile
import unittest
from pathlib import Path

from checks.screening_v3 import read_jsonl, write_jsonl
from script.analyze_adapter_strength_v1_judges import (
    ROOT, build_context, find_new_runs, validate_new_scores, verify_attempt_history,
)


class AdapterStrengthJudgeRetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = build_context()
        cls.source = ROOT / 'results/adapter_strength_v1/judges/incoming'

    def test_real_trace_preserves_retry_source_and_all_calls(self):
        runs = find_new_runs(self.source)
        rows, _ = validate_new_scores(self.context, self.source, runs)
        retried = [row for row in rows['qwen'] if row['source_run_id'] != runs['qwen']['id']]
        self.assertEqual(len(retried), 1)
        self.assertEqual(retried[0]['source_run_id'], runs['qwen']['retry_runs'][0]['id'])
        audit = verify_attempt_history(self.context, self.source, runs)
        self.assertEqual(audit['api_calls'], 33)
        self.assertEqual(audit['invalid_attempts'], 1)

    def test_parent_run_cannot_be_substituted_for_actual_retry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'incoming'
            shutil.copytree(self.source, target)
            runs = find_new_runs(target)
            rows = read_jsonl(target / 'qwen_results.jsonl')
            retry_id = runs['qwen']['retry_runs'][0]['id']
            next(row for row in rows if row['run_id'] == retry_id)['run_id'] = runs['qwen']['id']
            write_jsonl(target / 'qwen_results.jsonl', rows)
            with self.assertRaisesRegex(ValueError, 'actual accepted attempt'):
                verify_attempt_history(self.context, target, runs)

    def test_valid_response_cannot_be_retried_for_a_better_score(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'incoming'
            shutil.copytree(self.source, target)
            runs = find_new_runs(target)
            initial = read_jsonl(target / 'raw/qwen_initial_output.jsonl')
            retry = read_jsonl(target / 'raw/qwen_retry1_output.jsonl')[0]
            next(row for row in initial if row['uid'] == retry['uid'])['model_response'] = retry['model_response']
            write_jsonl(target / 'raw/qwen_initial_output.jsonl', initial)
            with self.assertRaisesRegex(ValueError, 'valid prior result was retried'):
                verify_attempt_history(self.context, target, runs)


if __name__ == '__main__':
    unittest.main()
