import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ZepPolicyTest(unittest.TestCase):
    def test_native_prompt_and_portable_defaults(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        post_source = (ROOT / "generate_answer.py").read_text(encoding="utf-8")
        combined = main_source + post_source
        self.assertNotIn("benchmark_prompt", combined)
        self.assertIn("ANSWER_SYSTEM_PROMPT", main_source)
        self.assertIn("retrieval_top_k=10", main_source)
        self.assertNotIn("/home/", combined)
        self.assertNotIn("/Users/", combined)


if __name__ == "__main__":
    unittest.main()
