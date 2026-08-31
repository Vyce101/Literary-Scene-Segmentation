# Tests synthetic training serialization and completion-only masking without a GPU.

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from tools.training_preflight import (
    IGNORE_INDEX,
    build_synthetic_example,
    build_parser,
    selective_completion_loss,
    serialize_training_example,
)


class CharacterBackend:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return type("Encoded", (), {"ids": [ord(char) for char in text]})()


class CharacterTokenizer:
    def __init__(self) -> None:
        self._tokenizer = CharacterBackend()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        serialized = "".join(
            f"<{message['role']}>{message['content']}"
            for message in messages
        )
        if add_generation_prompt:
            serialized += "<assistant>"
        return serialized if not tokenize else [ord(char) for char in serialized]

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(char) for char in text]


class TrainingPreflightTests(unittest.TestCase):
    def test_selective_loss_uses_only_causal_predictions_for_supervised_targets(self) -> None:
        full_logits = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 3.0, 2.0, 1.0],
                    [1.0, 3.0, 0.0, 2.0],
                    [2.0, 0.0, 4.0, 1.0],
                    [3.0, 1.0, 2.0, 5.0],
                    [5.0, 2.0, 1.0, 0.0],
                ]
            ],
            requires_grad=True,
        )

        class SelectiveModel:
            def __init__(self) -> None:
                self.logits_to_keep = None
                self.received_labels = "unset"

            def __call__(self, **kwargs):
                self.logits_to_keep = kwargs["logits_to_keep"]
                self.received_labels = kwargs["labels"]
                logits = full_logits.index_select(1, self.logits_to_keep)
                return SimpleNamespace(logits=logits)

        model = SelectiveModel()
        labels = torch.tensor([[-100, -100, 2, -100, 3, -100]])
        loss = selective_completion_loss(
            model,
            torch.zeros((1, 6), dtype=torch.long),
            torch.ones((1, 6), dtype=torch.long),
            labels,
        )
        expected = torch.nn.functional.cross_entropy(
            full_logits[:, [1, 3], :].reshape(-1, 4),
            torch.tensor([2, 3]),
        )
        self.assertTrue(torch.equal(model.logits_to_keep, torch.tensor([1, 3])))
        self.assertIsNone(model.received_labels)
        self.assertTrue(torch.allclose(loss, expected))

    def test_attention_implementation_defaults_to_sdpa_and_is_configurable(self) -> None:
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["--method", "lora", "--sequence-length", "1024"])
            .attn_implementation,
            "sdpa",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "--method",
                    "full",
                    "--sequence-length",
                    "1024",
                    "--attn-implementation",
                    "eager",
                ]
            ).attn_implementation,
            "eager",
        )

    def test_masks_system_and_user_but_not_assistant_response(self) -> None:
        tokenizer = CharacterTokenizer()
        example = serialize_training_example(
            tokenizer,
            "frozen system",
            "PAST\n[TARGET] text\n\nFUTURE\n",
            '{"boundaries_before":[],"document_boundaries_before":[]}',
        )
        response_start = len(tokenizer._tokenizer.encode(
            "<system>frozen system<user>PAST\n[TARGET] text\n\nFUTURE\n<assistant>",
            add_special_tokens=False,
        ).ids)
        self.assertTrue(all(label == IGNORE_INDEX for label in example.labels[:response_start]))
        self.assertEqual(
            example.labels[response_start:],
            tuple(ord(char) for char in '{"boundaries_before":[],"document_boundaries_before":[]}'),
        )

    def test_synthetic_example_uses_shared_user_format_and_fits_requested_length(self) -> None:
        tokenizer = CharacterTokenizer()
        example = build_synthetic_example(tokenizer, "SYSTEM", 1800)
        self.assertLessEqual(example.serialized_token_count, 1800)
        self.assertIn("PAST\n", example.user_message)
        self.assertIn("TARGET\n", example.user_message)
        self.assertIn("FUTURE\n", example.user_message)
        self.assertTrue(example.target_unit_ids)
        self.assertEqual(
            len(example.input_ids),
            len(example.labels),
        )


if __name__ == "__main__":
    unittest.main()
