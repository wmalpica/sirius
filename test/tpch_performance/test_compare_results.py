"""Cross-frontend comparison must preserve integer/decimal correctness."""

import unittest
import tempfile
from pathlib import Path

from compare_results import ParsedResult, compare_values, with_json_rows


class CrossFrontendValues(unittest.TestCase):
    def test_json_reference_preserves_whitespace_and_decimal_precision(self):
        reference = ParsedResult(["text", "money"], ["VARCHAR", "DECIMAL(38,2)"], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text('[{"text":" padded ","money":123456789012345678.90}]')
            reference = with_json_rows(reference, path)
        self.assertTrue(
            compare_values(reference, [[" padded ", "123456789012345678.9000"]], 1e-6)[
                0
            ]
        )
        self.assertFalse(
            compare_values(reference, [["padded", "123456789012345678.90"]], 1e-6)[0]
        )

    def test_decimal_scale_and_integer_precision(self):
        reference = ParsedResult(
            ["money", "count"],
            ["DECIMAL(38,2)", "BIGINT"],
            [["123.40", "9007199254740993"]],
        )
        self.assertTrue(
            compare_values(reference, [["123.4000", "9007199254740993"]], 1e-6)[0]
        )
        self.assertFalse(
            compare_values(reference, [["123.40", "9007199254740992"]], 1e-6)[0]
        )
        self.assertFalse(
            compare_values(reference, [["123.41", "9007199254740993"]], 1e-6)[0]
        )

    def test_floating_tolerance_does_not_apply_to_text_or_nulls(self):
        reference = ParsedResult(
            ["f", "text", "n"],
            ["DOUBLE", "VARCHAR", "INTEGER"],
            [["1.0", "001", "NULL"]],
        )
        self.assertTrue(compare_values(reference, [[1.00000001, "001", None]], 1e-6)[0])
        self.assertFalse(compare_values(reference, [[1.0, "1", None]], 1e-6)[0])
        self.assertFalse(compare_values(reference, [[1.0, "001", 0]], 1e-6)[0])


if __name__ == "__main__":
    unittest.main()
