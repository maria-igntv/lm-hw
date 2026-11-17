"""Guard loss masking and token weighting in the tiny-training pipeline."""

import importlib.util
import unittest
from types import SimpleNamespace

from checks.sft_tiny_v3 import evaluate_model, train_tiny_model


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required for numerical checks')
class TinyTrainingNumericsTest(unittest.TestCase):
    def test_accumulation_weights_tokens_instead_of_examples(self):
        import torch

        class QuadraticLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_B = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(loss=(self.lora_B - input_ids[0, 0].float()) ** 2)

        model = QuadraticLoss()
        batches = [
            {'input_ids': [3, 0, 0, 0, 0], 'labels': [-100, -100, -100, -100, 5]},
            {'input_ids': [-1, 0, 0, 0, 0], 'labels': [-100, -100, 5, 5, 5]},
        ]
        settings = {'seed': 1, 'gradient_accumulation_steps': 2, 'optimizer_steps': 1,
                    'learning_rate': 0.01, 'weight_decay': 0.0, 'max_grad_norm': 100.0,
                    'evaluation_steps': []}
        history = []
        train_tiny_model(model, batches, ['one_token', 'three_tokens'], settings,
                         'cpu', torch.float32, history.append, lambda step: None)
        # The token-weighted gradient is (-6 * 1 + 2 * 3) / 4 = 0.
        # Equal example weights would move the parameter instead.
        self.assertEqual(model.lora_B.item(), 0.0)
        self.assertEqual(history[0]['supervised_tokens'], 4)
        self.assertEqual(history[0]['token_weighted_loss'], 3.0)

    def test_teacher_forced_accuracy_ignores_prompt_and_padding(self):
        import torch

        class FixedLogits(torch.nn.Module):
            def forward(self, **kwargs):
                logits = torch.zeros((1, 6, 10))
                for position, token_id in enumerate([0, 4, 9, 6, 0, 0]):
                    logits[0, position, token_id] = 10
                return SimpleNamespace(loss=torch.tensor(2.0), logits=logits)

        model = FixedLogits().train()
        batch = {'input_ids': [1, 2, 4, 5, 6, 0], 'attention_mask': [1, 1, 1, 1, 1, 0],
                 'labels': [-100, -100, 4, 5, 6, -100]}
        result = evaluate_model(model, [batch], ['case'], 'cpu', torch.float32)
        self.assertEqual(result['supervised_tokens'], 3)
        self.assertAlmostEqual(result['token_accuracy'], 2 / 3)
        self.assertEqual(result['eos_accuracy'], 1.0)
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
