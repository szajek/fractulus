import unittest
import dicttools


class FrozenDictTest(unittest.TestCase):
    def test_Keys_Always_ReturnKeysView(self):
        d = dicttools.FrozenDict({1: 1, 2: 2})
        self.assertEqual(
            {1, 2},
            set(d.keys())
        )