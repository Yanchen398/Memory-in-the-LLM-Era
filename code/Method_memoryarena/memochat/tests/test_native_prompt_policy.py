import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MemoChatPolicyTest(unittest.TestCase):
    def test_native_prompt_and_portable_defaults(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("benchmark_prompt", source)
        self.assertIn('prompts["chatting"]["system"]', source)
        self.assertIn("return [10]", source)
        self.assertNotIn("/home/", source)
        self.assertNotIn("/Users/", source)


if __name__ == "__main__":
    unittest.main()
