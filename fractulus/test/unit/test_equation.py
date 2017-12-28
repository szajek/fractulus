import math
import unittest

from fractulus.equation import (CaputoSettings, create_right_caputo_stencil, create_left_caputo_stencil, \
                                create_riesz_caputo_stencil)


class CaputoStencilTest(unittest.TestCase):
    def test_LeftSide_Always_ReturnStencilWithCorrectWeights(self):

        lf = 0.6
        resolution = 4
        alpha = 0.5

        settings = CaputoSettings(alpha=alpha, lf=lf, resolution=resolution)
        alpha = settings.alpha

        _operator = create_left_caputo_stencil(alpha, lf=lf, p=resolution, increase_order_by=0.)
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

        settings = CaputoSettings(alpha=alpha, lf=lf, resolution=resolution)
        alpha = settings.alpha

        _operator = create_right_caputo_stencil(alpha, lf=lf, p=resolution, increase_order_by=0.)
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
        _settings = CaputoSettings(alpha, lf, resolution)

        _stencil = create_riesz_caputo_stencil(_settings)
        result = _stencil.expand(0.).weights

        self.assertAlmostEqual(1., result[0], places=5)
        self.assertEqual(1., sum(result.values()))

    @staticmethod
    def _calc_expected_weights(settings, left_limit, right_limit, u_0_weight_provider, u_p_weight_provider,
                               u_j_weight_provider, multiplier_provider):

        alpha = settings.alpha
        lf = settings.lf
        resolution = p = settings.resolution
        delta = lf/resolution

        n = math.floor(alpha) + 1.
        multiplier = multiplier_provider(n) * 1. / math.gamma(n - alpha + 2.)
        index = (n - alpha + 1.)
        u_0_weight = u_0_weight_provider(p, n, index)
        u_p_weight = u_p_weight_provider(p, n, index)

        weights = {
            -left_limit: u_0_weight,
            right_limit: u_p_weight,
        }
        for j in range(1, p):
            relative_node_address = -left_limit + j * delta
            weights[relative_node_address] = u_j_weight_provider(p, j, index)
        return {relative_address: multiplier * weight for relative_address, weight in weights.items()}
