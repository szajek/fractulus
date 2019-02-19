import math
import unittest

from fdm import Scheme, Stencil
from fdm.geometry import Point
from fractulus.equation import (Settings, create_right_caputo_stencil, create_left_caputo_stencil, \
                                create_riesz_caputo_stencil, create_left_rectangle_rule_stencil,
                                create_right_rectangle_rule_stencil, create_left_trapezoidal_rule_stencil,
                                create_right_trapezoidal_rule_stencil)


class CaputoStencilTest(unittest.TestCase):
    def test_LeftSide_Always_ReturnStencilWithCorrectWeights(self):

        lf = 0.6
        resolution = 4
        alpha = 0.5

        settings = Settings(alpha=alpha, lf=lf, resolution=resolution)
        alpha = settings.alpha

        _operator = create_left_caputo_stencil(alpha, lf=lf, p=resolution)
        results = _operator._weights

        expected = self._calc_expected_weights(
            settings,
            settings.lf,
            0.,
            lambda p, n, index: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
            lambda p, n, index: 1.,
            lambda p, j, index: (p - j + 1.) ** index - 2. * (p - j) ** index + (p - j - 1.) ** index,
            lambda n: 1.,
        )
        self.assertEqual(expected, results)

    def test_RightSide_Always_ReturnStencilWithCorrectWeights(self):
        resolution = lf = 4
        alpha = 0.5

        settings = Settings(alpha=alpha, lf=lf, resolution=resolution)
        alpha = settings.alpha

        _operator = create_right_caputo_stencil(alpha, lf=lf, p=resolution)
        results = _operator._weights

        expected = self._calc_expected_weights(
            settings,
            0.,
            settings.lf,
            lambda p, n, index: 1.,
            lambda p, n, index: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
            lambda p, j, index: (j + 1.) ** index - 2. * j ** index + (j - 1.) ** index,
            lambda n: (-1.)**n
        )

        self.assertEqual(expected, results)

    def test_Expand_AlphaAlmostOne_ReturnOnlyOneWeightForGivenNodeAddress(self):
        alpha = 0.999999
        lf = 1.
        resolution = 1
        _settings = Settings(alpha, lf, resolution)

        _stencil = create_riesz_caputo_stencil(_settings)
        result = _stencil.expand(Point(0.))

        self.assertAlmostEqual(1., result[Point(0)].real, places=5)
        self.assertAlmostEqual(1., sum([v.real for v in result.values()]), places=5)

    @staticmethod
    def _calc_expected_weights(settings, left_limit, right_limit, u_0_weight_provider, u_p_weight_provider,
                               u_j_weight_provider, multiplier_provider):

        alpha = settings.alpha
        lf = settings.lf
        resolution = p = settings.resolution
        h = lf/resolution
        n = math.floor(alpha) + 1.

        multiplier = multiplier_provider(n) * (h ** (n - alpha)) / math.gamma(n - alpha + 2.)
        index = (n - alpha + 1.)
        u_0_weight = u_0_weight_provider(p, n, index)
        u_p_weight = u_p_weight_provider(p, n, index)

        weights = {
            -left_limit: u_0_weight,
            right_limit: u_p_weight,
        }
        for j in range(1, p):
            relative_node_address = -left_limit + j * h
            weights[relative_node_address] = u_j_weight_provider(p, j, index)
        return {Point(relative_address): multiplier * weight for relative_address, weight in weights.items()}


class RectangleRuleStencilTest(unittest.TestCase):
    def test_Create_Left_ReturnCorrectStencil(self):

        result = create_left_rectangle_rule_stencil(Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(-0.6): 0.1352142643344008,
            Point(-0.4): 0.16038909801255471,
            Point(-0.2): 0.20902314205707648,
            Point(-0.0): 0.504626504404032
        })

        self.assertEqual(expected, result)

    def test_Create_Right_ReturnCorrectStencil(self):

        result = create_right_rectangle_rule_stencil(Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(0.): -0.504626504404032,
            Point(0.2): -0.20902314205707648,
            Point(0.4): -0.16038909801255471,
            Point(0.6): -0.1352142643344008
        })

        self.assertEqual(expected, result)


class TrapezoidalRuleStencilTest(unittest.TestCase):
    def test_Create_Left_ReturnCorrectStencil(self):

        result = create_left_trapezoidal_rule_stencil(Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(-0.8): 0.06598914093388651,
            Point(-0.6): 0.14671924087499555,
            Point(-0.4): 0.1814294346537252,
            Point(-0.2): 0.2786975227427686,
            Point(-0.0): 0.33641766960268793
        })

        self.assertEqual(expected, result)

    def _test_Create_Right_ReturnCorrectStencil(self):

        result = create_right_trapezoidal_rule_stencil(Settings(alpha=0.5, lf=0.8, resolution=4))

        expected = Stencil({
            Point(0.): -0.33641766960268793,
            Point(0.2): -0.2786975227427686,
            Point(0.4): -0.1814294346537252,
            Point(0.6): -0.14671924087499555,
            Point(0.8): -0.06598914093388651,
        })

        self.assertEqual(expected, result)