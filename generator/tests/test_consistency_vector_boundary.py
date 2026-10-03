import ast
import unittest
from pathlib import Path


class ConsistencyVectorBoundaryTest(unittest.TestCase):
    def test_model_facing_vector_has_only_image_available_keys(self):
        source = Path(__file__).resolve().parents[1] / "src" / "build_dataset.py"
        tree = ast.parse(source.read_text())
        dictionaries = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
        consistency = next(node for node in dictionaries if any(isinstance(key, ast.Constant) and key.value == "qr_readable" for key in node.keys))
        keys = {key.value for key in consistency.keys if isinstance(key, ast.Constant)}
        self.assertEqual(keys, {"qr_readable", "text_qr_match_score", "checksum_valid", "format_valid"})
        forbidden = {"expected_consistency_vector", "tamper_type", "final_label", "cnn_label", "ground_truth_fields", "record_id"}
        self.assertFalse(keys & forbidden)


if __name__ == "__main__":
    unittest.main()
