import math
import os
import unittest

from transformers import AutoTokenizer

from core import PromptRenderer, normalized_entropy_confidence


MODEL_PATH = os.environ.get(
    "JEV_MODEL_PATH",
    "Qwen/Qwen3.5-2B",
)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=os.path.isdir(os.path.expanduser(MODEL_PATH)),
        )

    def test_chat_candidates_extend_prompt_by_one_token(self):
        decision = PromptRenderer(self.tokenizer, "chat").render(
            "state", "question", (("x", "first"), ("y", "second"))
        )
        self.assertEqual(decision.candidate_tokens, ("A", "B"))
        self.assertEqual(decision.candidate_token_ids, (32, 33))
        self.assertTrue(decision.prompt.endswith("</think>\n\n"))

    def test_raw_candidates_use_leading_space_tokens(self):
        decision = PromptRenderer(self.tokenizer, "raw").render(
            "state", "question", (("x", "first"), ("y", "second"))
        )
        self.assertEqual(decision.candidate_tokens, (" A", " B"))
        self.assertEqual(decision.candidate_token_ids, (357, 417))
        self.assertTrue(decision.prompt.endswith("Answer:"))

    def test_normalized_entropy_confidence(self):
        self.assertAlmostEqual(normalized_entropy_confidence((0.5, 0.5)), 0.0)
        self.assertAlmostEqual(normalized_entropy_confidence((1.0, 0.0)), 1.0)
        expected = 1.0 - (-(0.75 * math.log(0.75) + 0.25 * math.log(0.25))) / math.log(2)
        self.assertAlmostEqual(
            normalized_entropy_confidence((0.75, 0.25)), expected
        )


if __name__ == "__main__":
    unittest.main()
