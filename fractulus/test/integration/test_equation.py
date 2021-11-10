from fdm.geometry import Point
from fractulus.equation import Settings, create_left_stencil
import unittest
import numpy as np

import fractulus as fr
import fdm


class LeftCaputoStencilTest(unittest.TestCase):
    def test_Expand_NodeZeroAndAlphaAlmostOne_ReturnNodeZeroWeightAlmostOne(self):

        stencil = create_left_stencil('caputo', Settings(0.9999, 10., 10))

        self.assertAlmostEqual(
            1.,
            stencil.expand(Point(0))[Point(0)],
            places=3,
        )


class LeftCaputoForLinearFunctionStudies(unittest.TestCase):
    def setUp(self):
        self._function = lambda point: point.x  # f(x) = x

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
        return self._compute_by_stencil(stencil.expand(Point(i)), self._function)

    @staticmethod
    def _compute_by_stencil(scheme, function):
        return sum([function(node) * weight for node, weight in scheme.items()])

    @staticmethod
    def _create_derivative(alpha, lf, p):
        return fdm.Operator(
            fr.equation.create_left_stencil('caputo', fr.equation.Settings(alpha, lf, p)),
            fdm.Operator(
                fdm.Stencil.central(.1)
            )
        )


class RieszCaputoStudy(unittest.TestCase):

    def setUp(self):
        raise NotImplementedError

    def _compute(self, alpha, lf, test_range):
        return [self._compute_for_item(i, alpha, lf) for i in test_range]

    def _compute_for_item(self, i, alpha, lf):
        stencil = self._create_derivative(alpha, lf)
        scheme = stencil.expand(Point(i))

        return self._compute_by_scheme(scheme)

    @staticmethod
    def _compute_by_scheme(scheme):
        return sum(scheme.values())

    def _create_derivative(self, alpha, lf):
        return fdm.Operator(
            fr.create_riesz_caputo_stencil(
                'caputo',
                fr.Settings(alpha, lf, lf)
            ),
            fdm.Number(self._function)
        )


class RieszCaputoForLinearDerivativeFunctionStudy(RieszCaputoStudy):
    def setUp(self):
        self._function = lambda point: point.x  # f'(x) = x

    def test_Calculate_AlphaAlmostOne_ValueEqualsClassicalDerivative(self):

        alpha = 0.99999
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range)

        expected = [i for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)

    def test_Calculate_AlphaAlmostZero_ValueEqualsClassicalDerivativeMultipliedByLengthScale(self):

        alpha = 0.000000000001
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range)

        expected = [i*lf for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=2)

    def test_Calculate_AlphaBetweenZeroAndOne_ValueEqualsClassicalDerivativeMultipliedByFactor(self):

        alpha = 0.8
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range)

        factor = lf**(1.-alpha)
        expected = [i*factor for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)


class RieszCaputoForQuadraticDerivativeFunctionStudy(RieszCaputoStudy):
    def setUp(self):
        self._function = lambda point: point.x**2  # f'(x) = x**2

    def test_Calculate_AlphaAlmostOne_ValueEqualsClassicalDerivative(self):

        alpha = 0.999999
        lf = 4
        test_range = range(0, 15)

        result = self._compute(alpha, lf, test_range)

        expected = [i**2 for i in test_range]

        np.testing.assert_almost_equal(expected, result, decimal=3)

