from fractulus.equation import CaputoSettings, create_left_caputo_stencil
import unittest
import numpy as np

import fractulus as fr
import fdm


class LeftCaputoStencilTest(unittest.TestCase):
    def test_Expand_NodeZeroAndAlphaAlmostOne_ReturnNodeZeroWeightAlmostOne(self):

        stencil = create_left_caputo_stencil(0.9999, 10., 10, 0.)

        self.assertAlmostEqual(
            1.,
            stencil.expand(0)._weights[0],
            places=3,
        )


class LeftCaputoForLinearFunctionStudies(unittest.TestCase):
    def setUp(self):
        self._function = lambda x: x  # f(x) = x

    def test_Calculate_AlphaAlmostOne_ConstantValue(self):

        alpha = 0.9999
        test_range = range(1, 15)

        result = self._compute(alpha, test_range)

        expected = [1. for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)

    def test_Calculate_AlphaAlmostZero_ReturnLinearFunction(self):

        alpha = 0.00001
        test_range = range(1, 2)

        result = self._compute(alpha, test_range)

        expected = test_range

        np.testing.assert_almost_equal(expected, result, decimal=3)

    def _compute(self, alpha, test_range):
        return [self._compute_for_item(i, alpha) for i in test_range]

    def _compute_for_item(self, i, alpha):
        stencil = self._create_derivative(alpha, i, i)
        return self._compute_by_stencil(stencil.expand(i), self._function)

    @staticmethod
    def _compute_by_stencil(scheme, function):
        coefficients = scheme.to_coefficients(1.)
        return sum([function(node) * weight for node, weight in coefficients.items()])

    @staticmethod
    def _create_derivative(alpha, lf, p):
        return fdm.Operator(
            fr.equation.create_left_caputo_stencil(alpha, lf, p, 0.),
            fdm.Operator(
                fdm.Stencil.central(.1)
            )
        )


class RieszCaputoStudy(unittest.TestCase):

    def setUp(self):
        raise NotImplementedError

    def _compute(self, alpha, lf, test_range, h=2):
        return [self._compute_for_item(i, alpha, lf, h) for i in test_range]

    def _compute_for_item(self, i, alpha, lf, h):
        stencil = self._create_derivative(alpha, lf)
        scheme = stencil.expand(i)

        return self._compute_by_stencil(scheme, h)

    @staticmethod
    def _compute_by_stencil(scheme, h):
        coefficients = scheme.to_coefficients(h)
        return sum(coefficients.values())

    def _create_derivative(self, alpha, lf):
        return fdm.Operator(
            fr.create_riesz_caputo_stencil(
                fr.CaputoSettings(alpha, lf, lf)
            ),
            fdm.Number(self._function)
        )


class RieszCaputoForLinearDerivativeFunctionStudy(RieszCaputoStudy):
    def setUp(self):
        self._function = lambda x: x  # f'(x) = x

    def test_Calculate_AlphaAlmostOne_ValueEqualsClassicalDerivative(self):

        alpha = 0.99999
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range, h=1.)

        expected = [i for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)

    def test_Calculate_AlphaAlmostZero_ValueEqualsClassicalDerivativeMultipliedByLengthScale(self):

        alpha = 0.000000000001
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range, h=1)

        expected = [i*lf for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=2)

    def test_Calculate_AlphaBetweenZeroAndOne_ValueEqualsClassicalDerivativeMultipliedByFactor(self):

        alpha = 0.8
        lf = p = 4
        h = 2.
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range, h=2.)

        factor = (p*h)**(1.-alpha)
        expected = [i*factor for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)


class RieszCaputoForQuadraticDerivativeFunctionStudy(RieszCaputoStudy):
    def setUp(self):
        self._function = lambda x: x**2  # f'(x) = x**2

    def test_Calculate_AlphaAlmostOne_ValueEqualsClassicalDerivative(self):

        alpha = 0.999999
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range)

        expected = [i**2 for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)

