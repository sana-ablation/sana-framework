import unittest

from sana_evaluation.helper.metrics import compute_efficiency_metrics


class EfficiencyMetricsTests(unittest.TestCase):
    def test_no_queries_with_expected_queries_has_zero_efficiency(self):
        result = compute_efficiency_metrics([], ["expected"], 1)

        self.assertEqual(result["query_count"], 0)
        self.assertEqual(result["query_efficiency"], 0.0)

    def test_no_queries_when_none_are_needed_has_full_efficiency(self):
        result = compute_efficiency_metrics([], [], 0)

        self.assertEqual(result["query_count"], 0)
        self.assertEqual(result["query_efficiency"], 1.0)


if __name__ == "__main__":
    unittest.main()
