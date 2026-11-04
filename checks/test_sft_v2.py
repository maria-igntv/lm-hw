"""Regression checks for scenario leakage, category coverage and audit provenance."""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SFTV2DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_jsonl(ROOT / "data/sft_v2/sft_train.jsonl")
        cls.val = read_jsonl(ROOT / "data/sft_v2/sft_val.jsonl")
        cls.rows = cls.train + cls.val

    def test_group_holdout_and_all_categories_in_both_splits(self):
        self.assertFalse({r["scenario_group"] for r in self.train} &
                         {r["scenario_group"] for r in self.val})
        train_categories = {r["category"] for r in self.train}
        self.assertEqual(len(train_categories), 8)
        self.assertEqual(train_categories, {r["category"] for r in self.val})
        split = {r["sample_id"]: name for name, rows in (("train", self.train), ("val", self.val)) for r in rows}
        for pair in read_jsonl(ROOT / "data/sft_v2/similar_question_pairs.jsonl"):
            self.assertEqual(split[pair["left"]], split[pair["right"]])

    def test_audit_statuses_and_revisions_match_evidence(self):
        decisions = {r["sample_id"]: r for r in read_jsonl(ROOT / "data/audit_v2/decisions.jsonl")}
        self.assertEqual(len(decisions), 48)
        counts = Counter(r["audit_status"] for r in self.rows)
        self.assertEqual(counts, dict(keep=7, revise=41, not_reviewed=602))
        self.assertEqual(len({r["sample_id"] for r in self.rows}), 650)
        for row in self.rows:
            if row["sample_id"] in decisions:
                self.assertEqual(row["messages"][2]["content"], decisions[row["sample_id"]]["revised_answer"])

    def test_manifest_hashes_and_uniform_chat_format(self):
        manifest = json.loads((ROOT / "data/sft_v2/manifest.json").read_text())
        for name, expected in manifest["output_hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT / "data/sft_v2" / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(len({r["messages"][0]["content"] for r in self.rows}), 1)
        for row in self.rows:
            self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
