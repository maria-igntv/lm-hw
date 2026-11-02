import unittest

from checks.evaluation import compare_answers, validate_answers


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.q = [dict(id=i, question=f'Газ: вопрос {i}', category='gas', dangerous=True)
                  for i in range(1, 5)]
        good = 'Это опасно. Вызовите специалиста для проверки оборудования.'
        bad = 'Просто продолжайте пользоваться оборудованием.'
        self.before = [{**q, 'answer': a} for q, a in zip(self.q, [bad, good, bad, good])]
        self.after = [{**q, 'answer': a} for q, a in zip(self.q, [good, bad, bad, good])]

    def test_pairing_uses_ids_not_row_order(self):
        scores, changes, _ = compare_answers(self.q, self.before, self.after[::-1])
        self.assertEqual(changes, dict(fixed=1, regressed=1, still_missing=1, both_pass=1))
        self.assertEqual(scores['baseline']['safety_pass_rate'], 0.5)

    def test_rejects_changed_question_or_label(self):
        for field, value in [('question', 'Другой вопрос'), ('dangerous', False)]:
            rows = [dict(r) for r in self.after]
            rows[0][field] = value
            with self.assertRaises(ValueError):
                compare_answers(self.q, self.before, rows)

    def test_rejects_duplicates_and_missing_answers(self):
        for rows in [self.after[:-1], self.after + self.after[:1]]:
            with self.assertRaises(ValueError):
                validate_answers(self.q, rows)
        self.assertEqual(len(validate_answers(self.q, self.after[:1], complete=False)), 1)


if __name__ == '__main__':
    unittest.main()
