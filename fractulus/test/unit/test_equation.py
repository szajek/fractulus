import unittest

from fdm import Stencil
from fdm.geometry import Point
from fractulus.equation import (Settings, create_left_stencil, create_riesz_caputo_stencil)


class CaputoStencilTest(unittest.TestCase):
    def test_LeftSide_Always_ReturnStencilWithCorrectWeights(self):
        lf = 0.6
        resolution = 4
        alpha = 0.5

        settings = Settings(alpha=alpha, lf=lf, resolution=resolution)
        alpha = settings.alpha

        _operator = create_left_stencil('caputo', settings)
        results = _operator._weights

        expected = {
            Point(-0.44999999999999996, 0.0, 0.0): 0.12706258982171437,
            Point(-0.6, 0.0, 0.0): 0.057148272422657305,
            Point(0.0, 0.0, 0.0): 0.29134624815788773,
            Point(-0.15000000000000002, 0.0, 0.0): 0.24135913466702896,
            Point(-0.3, 0.0, 0.0): 0.1571224994043748
        }

        self.assertEqual(expected, results)

    def test_Expand_AlphaAlmostOne_ReturnOnlyOneWeightForGivenNodeAddress(self):
        alpha = 0.999999
        lf = 1.
        resolution = 1
        _settings = Settings(alpha, lf, resolution)

        _stencil = create_riesz_caputo_stencil('caputo', _settings)
        result = _stencil.expand(Point(0.))

        self.assertAlmostEqual(1., result[Point(0)].real, places=5)
        self.assertAlmostEqual(1., sum([v.real for v in result.values()]), places=5)


class RectangleRuleStencilTest(unittest.TestCase):
    def test_Create_Left_ReturnCorrectStencil(self):
        result = create_left_stencil('rectangle', Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(-0.6): 0.1352142643344008,
            Point(-0.4): 0.16038909801255471,
            Point(-0.2): 0.20902314205707648,
            Point(-0.0): 0.504626504404032
        })

        self.assertEqual(expected, result)


class TrapezoidalRuleStencilTest(unittest.TestCase):
    def test_Create_Left_ReturnCorrectStencil(self):
        result = create_left_stencil('trapezoidal', Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(-0.8): 0.06598914093388651,
            Point(-0.6): 0.14671924087499555,
            Point(-0.4): 0.1814294346537252,
            Point(-0.2): 0.2786975227427686,
            Point(-0.0): 0.33641766960268793
        })

        self.assertEqual(expected, result)


class SimpsonRuleStencilTest(unittest.TestCase):
    def test_Create_LeftEvenResolution_ReturnCorrectStencil(self):
        result = create_left_stencil('simpson', Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(-0.8): 0.04139442395762801,
            Point(-0.6): 0.19590867482751353,
            Point(-0.4): 0.10587690665922159,
            Point(-0.2): 0.38061314477925734,
            Point(-0.0): 0.28545985858444356
        })

        self.assertEqual(expected, result)

    def test_Create_LeftOddResolution_ReturnCorrectStencil(self):
        result = create_left_stencil('simpson', Settings(alpha=0.5, lf=1.0, resolution=5))

        expected = Stencil({
            Point(-1.0): 0.03727938104424598,
            Point(-0.8): 0.1690131707314828,
            Point(-0.6): 0.09488746599316666,
            Point(-0.4): 0.24273847930859488,
            Point(-0.2): 0.214401233455065,
            Point(-0.0): 0.40370120352322564,
            Point(0.2): -0.033641766960268805,
        })

        self.assertEqual(expected, result)

