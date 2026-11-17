"""Protect the EOS target and prompt boundary when building smoke labels."""

import unittest

from checks.sft_smoke_v3 import build_masked_example


class TokenStream:
    eos_token_id = 102
    pad_token_id = 102

    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 2:
            return [101, 20, 102, 30, 102, 40]
        return [101, 20, 102, 30, 102, 40, 50, 51, 102, 198]

    def decode(self, token_ids, **kwargs):
        return str(token_ids)


class SFTSmokeLabelsTest(unittest.TestCase):
    def setUp(self):
        self.record = {'sample_id': 'control', 'messages': [
            {'role': 'system', 'content': 'System'}, {'role': 'user', 'content': 'Question'},
            {'role': 'assistant', 'content': 'Answer'},
        ]}

    def test_eos_stays_supervised_when_padding_uses_the_same_token(self):
        batch, audit = build_masked_example(TokenStream(), self.record, 12)
        self.assertEqual(batch['labels'], [-100] * 6 + [50, 51, 102] + [-100] * 3)
        self.assertEqual(batch['attention_mask'], [1] * 9 + [0] * 3)
        self.assertEqual(audit['supervised_tokens'], 3)
        self.assertEqual(audit['last_supervised_token_id'], 102)
        self.assertEqual(audit['unpadded_tokens'], 9)

    def test_bpe_boundary_mismatch_aborts_label_construction(self):
        class DifferentPrefix(TokenStream):
            def apply_chat_template(self, messages, **kwargs):
                tokens = super().apply_chat_template(messages, **kwargs)
                if len(messages) == 3:
                    tokens[5] = 999
                return tokens

        with self.assertRaisesRegex(ValueError, 'boundary differs'):
            build_masked_example(DifferentPrefix(), self.record, 12)

    def test_oversized_control_is_rejected_without_truncating_target(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            build_masked_example(TokenStream(), self.record, 8)


if __name__ == '__main__':
    unittest.main()
