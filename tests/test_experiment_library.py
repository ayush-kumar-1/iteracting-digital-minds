"""Regression tests for lazy composition of the pilot materials."""

import unittest

from src.python.experiment_library.composition import compose_experiment
from src.python.experiment_library.validation import validate_library


class ExperimentLibraryTests(unittest.TestCase):
    """Verify structural QA and role-preserving, order-reversible composition."""

    def test_library_passes_structural_qa(self) -> None:
        self.assertEqual(validate_library(), [])

    def test_composition_preserves_roles_and_reverses_options(self) -> None:
        condition = {
            "frame_id": "F04",
            "history_length": 5,
            "profile_id": "P003",
            "scenario_id": "WVS_Q90_S04",
            "elicitation_id": "E04",
            "language": "en",
            "option_order": "BA",
        }
        result = compose_experiment(condition)
        messages = result["messages"]
        self.assertEqual(result["material_ids"]["history_id"], "F04_H5_R01")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("No preference", messages[-1]["content"])
        self.assertLess(
            messages[-1]["content"].index("Option A: The council would require a regional body"),
            messages[-1]["content"].index("Option B: The council would authorize a regional body"),
        )
        self.assertTrue(any(message["role"] == "developer" for message in messages))

    def test_rejects_missing_or_unsupported_conditions(self) -> None:
        with self.assertRaises(ValueError):
            compose_experiment({})
        with self.assertRaisesRegex(ValueError, "English"):
            compose_experiment({
                "frame_id": "F01", "history_length": 1, "profile_id": "P001",
                "scenario_id": "WVS_Q8_S01", "elicitation_id": "E01", "language": "es",
                "option_order": "AB",
            })


if __name__ == "__main__":
    unittest.main()
