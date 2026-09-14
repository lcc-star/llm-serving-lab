import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'study/stage05_correctness_attribution/inventory.py'
spec = importlib.util.spec_from_file_location('inventory', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DifferenceTests(unittest.TestCase):
    def test_equal_outputs(self):
        self.assertIsNone(module.first_difference([1, 2], [1, 2]))

    def test_first_and_later_divergence(self):
        self.assertEqual(module.first_difference([1, 2], [3, 2]), 0)
        self.assertEqual(module.first_difference([1, 2, 3], [1, 4, 3]), 1)

    def test_length_difference_is_not_silently_ignored(self):
        self.assertEqual(module.first_difference([1], [1, 2]), 1)
        self.assertEqual(module.first_difference([], [1]), 0)
