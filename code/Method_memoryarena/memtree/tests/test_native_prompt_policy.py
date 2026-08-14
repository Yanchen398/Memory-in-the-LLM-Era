import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MemTreePolicyTest(unittest.TestCase):
    def test_native_prompt_and_portable_defaults(self):
        utils_source = (ROOT / "utils.py").read_text(encoding="utf-8")
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        mp_source = (ROOT / "main_mp.py").read_text(encoding="utf-8")
        combined = utils_source + main_source + mp_source
        self.assertNotIn("benchmark_prompt", combined)
        self.assertIn("ANSWER_PROMPT.format", utils_source)
        self.assertIn("return [10]", mp_source)
        self.assertNotIn("/home/", combined)
        self.assertNotIn("/Users/", combined)


if __name__ == "__main__":
    unittest.main()
