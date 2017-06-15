import unittest

from finite_difference import Operator, Stencil, Number, Scheme, LazyOperation


class OperatorTest(unittest.TestCase):
    def test_FirstOrder_CentralDiffForNodeZero_GenerateProperCoefficients(self):
        value = 2.

        linear_operator = Operator(
            Stencil.central(),
            Number(value)
        )

        scheme = linear_operator.expand(0)

        self.assertEqual(
            Scheme({-1: -0.5, 1: 0.5}) * 2,
            scheme,
        )

    def test_FirstOrder_CentralDiffForNodeThree_GenerateProperCoefficients(self):
        value = 2.

        linear_operator = Operator(
            Stencil.central(),
            Number(value)
        )

        scheme = linear_operator.expand(3)

        self.assertEqual(
            Scheme({2: -0.5, 4: 0.5}) * 2,
            scheme,
        )

    def test_SecondOrder_CentralDiffForNodeThree_GenerateProperCoefficients(self):
        value = 2.

        linear_operator = Operator(
            Stencil.central(1.),
            Operator(
                Stencil.central(1.),
                Number(value)
            )
        )

        scheme = linear_operator.expand(3)

        self.assertEqual(
            Scheme({2: 1., 3: -2., 4: 1.}, order=2.) * 2,
            scheme,
        )


class LazyOperationTest(unittest.TestCase):
    def test_Summation_Schemes_ReturnSchemeWithProperWeights(self):

        w1 = 9.99995772128789e-06
        w2 = 0.99999577213334

        s1 = Stencil({-0.5: w1, 0.0: w2})
        s2 = Stencil({-0.5: w1, 0.0: w2})

        s = LazyOperation.summation(s1, s2)

        expected = Scheme({-0.5: w1*2, 0.0: w2*2})
        result = s.expand(0)

        self.assertEqual(expected, result)