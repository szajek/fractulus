import unittest

import math


class GammaFunction(unittest.TestCase):
    def test_Call_Always_ReturnFactorialFunctionOutputWithArgumentShiftDownBy1(self):
        self.assertEqual(1., math.gamma(1.))
        self.assertEqual(1., math.gamma(2.))
        self.assertEqual(2., math.gamma(3.))
        self.assertEqual(6., math.gamma(4.))
