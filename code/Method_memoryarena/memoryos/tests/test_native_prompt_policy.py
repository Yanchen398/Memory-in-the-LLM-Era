import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MemoryOSPolicyTest(unittest.TestCase):
    def test_native_prompts_and_top_k_ten(self):
        files = [
            ROOT / "main.py",
            ROOT / "main_lme.py",
            ROOT / "main_lme_runner.py",
        ]
        sources = {path.name: path.read_text(encoding="utf-8") for path in files}
        combined = "".join(sources.values())
        self.assertNotIn("benchmark_prompt", combined)
        self.assertIn("system_prompt = (", sources["main.py"])
        self.assertIn("memo.get_response(query=question)", sources["main_lme.py"])
        self.assertIn("memo.get_response(query=question)", sources["main_lme_runner.py"])
        self.assertIn("DEFAULT_RETRIEVAL_TOP_K = 10", combined)
        self.assertNotIn("/home/", combined)
        self.assertNotIn("/Users/", combined)


if __name__ == "__main__":
    unittest.main()
