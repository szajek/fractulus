import math
import unittest
from mock import MagicMock, patch

from fractulus.finite_difference import (LazyOperation, Operator, Stencil, Scheme, Number, Element, \
    Delta, NodeFunction, operate, merge_weights, Coefficients, MutateMixin)


def _are_the_same_objects(obj, mutated):
    return obj is mutated


class MutateMixinTest(unittest.TestCase):

    class PublicFieldsExample(MutateMixin):
        def __init__(self, field_1, field_2):
            self.field_1 = field_1
            self.field_2 = field_2

            MutateMixin.__init__(self, 'field_1', 'field_2')

    class PrivateFieldsExample(MutateMixin):
        def __init__(self, field_1, field_2):

            self._field_1 = field_1
            self._field_2 = field_2

            MutateMixin.__init__(self, ('field_1', '_field_1'), ('field_2', '_field_2'))

    def test_Mutate_PublicField_ReturnNewObjectWithMutatedField(self):
        obj = self.PublicFieldsExample('field_1_value', 'field_2_value')

        mutated = obj.mutate(field_1='new')

        self.assertFalse(_are_the_same_objects(obj, mutated))
        self.assertEqual('new', mutated.field_1)
        self.assertEqual('field_2_value', mutated.field_2)

    def test_Mutate_PrivateField_ReturnNewObjectWithMutatedField(self):
        obj = self.PrivateFieldsExample('field_1_value', 'field_2_value')

        mutated = obj.mutate(field_1='new')

        self.assertFalse(_are_the_same_objects(obj, mutated))
        self.assertEqual('new', mutated._field_1)
        self.assertEqual('field_2_value', mutated._field_2)


class DeltaTest(unittest.TestCase):
    def test_Average_Always_ReturnAverageValueForInputs(self):
        d = Delta(1., 2.)

        self.assertEquals(1.5, d.average)

    def test_MathOperations_Always_OperateOnAverageValueAndReturnFloat(self):
        d = Delta(1., 2.)
        avg = d.average

        self.assertEqual(avg + 1., d + 1.)
        self.assertEqual(avg * 2., d * 2.)
        self.assertEqual(avg - 2., d - 2.)
        self.assertEqual(avg / 2., d / 2.)


class CoefficientsTest(unittest.TestCase):

    def test_ToValue_Always_ReturnValueCalculatedBasedOnOutput(self):

        coefficients = Coefficients({1: 1.3, 2: 2.3})
        output = {1: 2.1, 2: 1.1}

        result = coefficients.to_value(output)

        self.assertEqual(
            1.3 * 2.1 + 2.3 * 1.1,
            result,
        )


class SchemeTest(unittest.TestCase):

    def test_Equal_Always_CompareDataAndOrder(self):
        self.assertEqual(
            self._build_scheme((1., 2., 3.)),
            self._build_scheme((1., 2., 3.))
        )
        self.assertNotEqual(
            self._build_scheme((1., 2., 3.)),
            self._build_scheme((1., 2., 4.))
        )
        self.assertNotEqual(
            self._build_scheme((1., 2., 3.)),
            self._build_scheme((1., 2.))
        )
        self.assertNotEqual(
            self._build_scheme((1., 2.), order=1.1),
            self._build_scheme((1., 2.))
        )

    def test_Duplicate_Always_ReturnClone(self):
        s1 = Scheme({1: 2, 3: 1.})

        result = s1.duplicate()

        self.assertEqual(s1, result)
        self.assertFalse(_are_the_same_objects(s1, result))

    def test_Add_None_ReturnUnchangedScheme(self):
        s1 = Scheme({1: 2, 3: 1.})
        s2 = None

        result = s1 + s2

        self.assertEqual(s1, result)

    def test_RightAdd_None_ReturnUnchangedScheme(self):
        s1 = None
        s2 = Scheme({1: 2, 3: 1.})

        result = s1 + s2

        self.assertEqual(s2, result)

    def test_Add_NoNodesSetsIntersection_MergeData(self):
        s1 = Scheme({1: 2})
        s2 = Scheme({2: 3})

        result = s1 + s2

        expected = Scheme({1: 2, 2: 3})

        self.assertEqual(expected, result)

    def test_Add_NodesSetsIntersection_MergeForUniqueAndAddForIntersection(self):
        s1 = Scheme({1: 2, 3: 1.})
        s2 = Scheme({2: 3, 3: 2.})

        result = s1 + s2

        expected = Scheme({1: 2, 2: 3, 3: 3})

        self.assertEqual(expected, result)

    def test_Add_InconsistentOrder_ThrowsAttributeException(self):

        s1 = Scheme({}, order=1)
        s2 = Scheme({}, order=2)

        with self.assertRaises(AttributeError):
            s1 + s2

    def test_Shift_Always_ShiftNodeAddresses(self):

        scheme = Scheme({0: 1., 3.: -3})

        result = scheme.shift(-1.)

        expected = Scheme({-1: 1., 2.: -3})

        self.assertEquals(result, expected)

    def test_LeftMultiplication_IntegerOrFloat_MultiplyWeights(self):

        scheme = self._build_scheme(weights=(0., 2., 3.))

        result = 2.*scheme

        expected = self._build_scheme(weights=(0., 4., 6.))

        self.assertEqual(expected, result)

    def test_RightMultiplication_IntegerOrFloat_MultiplyWeights(self):

        scheme = self._build_scheme(weights=(0., 2., 3.))

        result = scheme*2.

        expected = self._build_scheme(weights=(0., 4., 6.))

        self.assertEqual(expected, result)

    def test_ToCoefficients_Always_ReturnWeightsConsideringDeltaAndOrder(self):

        order = 2.
        weight = 1.
        delta = 2.
        scheme = Scheme({1: weight}, order)

        result = scheme.to_coefficients(delta)

        expected = {1: weight/delta**order}

        self.assertEquals(expected, result)

    def test_ToCoefficients_PositiveMidNode_SpreadEquallyWeightsToAdjacentNodes(self):

        scheme = Scheme({0.5: 1})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.5, 1: 0.5}

        self.assertEquals(expected, result)

    def test_ToCoefficients_NegativeMidNode_SpreadEquallyWeightsToAdjacentNodes(self):

        scheme = Scheme({-0.5: 1})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.5, -1: 0.5}

        self.assertEquals(expected, result)

    def test_ToCoefficients_NegativeFloatNodeAddress_SpreadProportionallyWeightsToAdjacentNodes(self):

        scheme = Scheme({-0.25: 1.})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.75, -1: 0.25}

        self.assertEquals(expected, result)

    def test_ToCoefficients_PositiveFloatNodeAddress_SpreadProportionallyWeightsToAdjacentNodes(self):
        scheme = Scheme({0.75: 1})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.25, 1: 0.75}

        self.assertEquals(expected, result)

    def _build_scheme(self, weights, order=1.):
        return Scheme(
            {i: d for i, d in enumerate(weights)},
            order
        )


class MergeWeightsTest(unittest.TestCase):
    def test_Call_NoIntersection_ReturnDictWithElementsFromBoth(self):

        weights_1 = {1: 1}
        weights_2 = {2: 2}

        result = merge_weights(weights_1, weights_2)

        expected = {1: 1, 2: 2}

        self.assertEqual(expected, result)

    def test_Call_TheSameNodeAddresses_ReturnDictWitSummedValues(self):
        weights_1 = {1: 1}
        weights_2 = {1: 2}

        result = merge_weights(weights_1, weights_2)

        expected = {1: 3}

        self.assertEqual(expected, result)

    def test_Call_Intersection_ReturnDictWithElementsFromBothAndSummedWeightsForSharedAddresses(self):
        weights_1 = {1: 1, 3: 1}
        weights_2 = {2: 2, 3: 2}

        result = merge_weights(weights_1, weights_2)

        expected = {1: 1, 2: 2, 3: 3}

        self.assertEqual(expected, result)

    def test_Call_ManyGiven_ReturnMerged(self):
        weights_1 = {1: 1}
        weights_2 = {2: 2}
        weights_3 = {3: 3}

        result = merge_weights(weights_1, weights_2, weights_3)

        expected = {1: 1, 2: 2, 3: 3}

        self.assertEqual(expected, result)


class OperateTest(unittest.TestCase):
    def test_Call_WithNone_ReturnTheSameScheme(self):
        scheme = Scheme({1: 2})

        result = operate(scheme, None)

        self.assertEqual(scheme, result)

    def test_Call_WithEmptySchemeOrElement_RaiseAttributeError(self):
        scheme = Scheme({1: 1})
        element = Stencil({1: 1})

        with self.assertRaises(AttributeError):
            operate(Scheme({}), element)

        with self.assertRaises(AttributeError):
            operate(scheme, Stencil({}))

    def test_Call_OneNodeScheme_ReturnRevolvedScheme(self):
        scheme = Scheme({1: 2})

        element = MagicMock(
            expand=lambda node_idx: Scheme({0: 1., 5.: 2.})
        )

        result = operate(scheme, element)

        expected = Scheme({0: 1. * 2., 5.: 2. * 2.}, order=2.)

        self.assertEqual(expected, result)

    def test_Call_WithFractionalScheme_ReturnSchemeWithSummedOrders(self):
        scheme = Scheme({1: 1}, order=1.2)

        element = Stencil({1: 1}, order=2.2)

        scheme = operate(scheme, element)

        expected = 3.4

        self.assertAlmostEqual(expected, scheme.order)

    def test_Call_WithSchemeVariedByNodeAddress_ReturnRevolvedScheme(self):
        scheme = Scheme({1: 2, 2: 4})

        element = MagicMock(
            expand=lambda node_idx: {
                1: Scheme({0: 1., 6.: 3.}),
                2: Scheme({0: 1., 5.: 2.}),
            }[node_idx]
        )

        result = operate(scheme, element)

        expected = Scheme({0: 1. * 2 + 1. * 4., 5.: 2. * 4., 6: 3. * 2.}, order=2.)

        self.assertEqual(expected, result)

    def test_Call_WithNumber_ReturnSchemeOfUnchangedOrder(self):

        scheme_order = 1.2
        scheme = Scheme({1: 1}, order=scheme_order)

        element = Number(0)

        revolved_scheme = operate(scheme, element)
        result = revolved_scheme.order

        expected = scheme_order

        self.assertAlmostEqual(expected, result)

#


class ElementTest(unittest.TestCase):
    class ConcreteElement(Element):
        def expand(self):
            return Scheme({1: 1})

    def test_Add_TwoElements_ReturnLazySummation(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 + e2

        expected = LazyOperation.summation(e1, e2)

        self.assertEquals(expected, result)

    def test_Multiplication_TwoElements_ReturnLazyMultiplication(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 * e2

        expected = LazyOperation.multiplication(e1, e2)

        self.assertEquals(expected, result)

    def test_Subtraction_TwoElements_ReturnLazySubtraction(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 - e2

        expected = LazyOperation.subtraction(e1, e2)

        self.assertEquals(expected, result)

    def test_Division_TwoElements_ReturnLazyDivision(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 / e2

        expected = LazyOperation.division(e1, e2)

        self.assertEquals(expected, result)


class StencilTest(unittest.TestCase):
    def test_Create_NoAxesProvided_AssignAxisOneByDefault(self):
        stencil = Stencil(1.)
        self.assertEqual(
            1,
            stencil._axis
        )

    def test_Expand_OrderDifferentThanOne_CreateSchemeWithGivenOrder(self):

        stencil_order = 1.2
        s = Stencil({}, order=stencil_order)

        expanded = s.expand(0.)
        order = expanded._order

        self.assertEqual(stencil_order, order)

    def test_Uniform_Always_GenerateUniformlyDistributedNodesInGivenLimits(self):
        left_range, right_range = 1., 2.
        resolution = 3.

        _stencil = Stencil.uniform(left_range, right_range, resolution, lambda *args: None)
        result = set(_stencil._weights.keys())

        _left_limit = -left_range
        _delta = (right_range + left_range) / resolution
        expected_node_addresses = set([_left_limit + _delta * i for i in range(4)])

        self.assertEquals(expected_node_addresses, result)

    def test_Central_RangeOne_GenerateWeightsEqualOneAssignedToAdjacentMidnodes(self):

        _scheme = Stencil.central(1.)
        result = _scheme._weights

        _scheme = Stencil({-0.5: -1., 0.5: 1.})
        expected = _scheme._weights

        self.assertTrue(self._compare_dict(expected, result))

    def test_Central_RangeTwo_GenerateWeightsEqualHalfAssignedToAdjacentNodes(self):

        _scheme = Stencil.central(2.)
        result = _scheme._weights

        _scheme = Stencil({-1.: -0.5, 1.: 0.5})
        expected = _scheme._weights

        self.assertTrue(self._compare_dict(expected, result,))

    def _compare_dict(self, d1, d2, tol=1e-4):
        return len(d1) == len(d2) and all(math.fabs(d1[k] - d2[k]) < tol for k in d1.keys())


class LazyOperationTest(unittest.TestCase):
    def test_Summation_Always_CallAddMagicForLeftAddendScheme(self):
        scheme_1 = MagicMock()
        scheme_2 = MagicMock()

        addend_1 = MagicMock(
            expand=MagicMock(return_value=scheme_1)
        )
        addend_2 = MagicMock(
            expand=MagicMock(return_value=scheme_2)
        )

        op = LazyOperation.summation(addend_1, addend_2)
        op.expand()

        scheme_1.__add__.assert_called_once()
        scheme_2.__add__.assert_not_called()

    def test_Subtraction_Always_CallSubMagicForMinuendScheme(self):
        scheme_minuend = MagicMock()
        scheme_subtrahend = MagicMock()

        minuend = MagicMock(
            expand=MagicMock(return_value=scheme_minuend)
        )
        subtrahend = MagicMock(
            expand=MagicMock(return_value=scheme_subtrahend)
        )

        op = LazyOperation.subtraction(minuend, subtrahend)
        op.expand()

        scheme_minuend.__sub__.assert_called_once()
        scheme_subtrahend.__sub__.assert_not_called()

    def test_Multiplication_Always_CallMultMagicForLeftFactorScheme(self):
        scheme_factor_1 = MagicMock()
        scheme_factor_2 = MagicMock()

        factor_1 = MagicMock(
            expand=MagicMock(return_value=scheme_factor_1)
        )
        factor_2 = MagicMock(
            expand=MagicMock(return_value=scheme_factor_2)
        )

        op = LazyOperation.multiplication(factor_1, factor_2)
        op.expand()

        scheme_factor_1.__mul__.assert_called_once()
        scheme_factor_2.__mul__.assert_not_called()

    def test_Division_Always_CallDivMagicForDividendScheme(self):
        scheme_dividend = MagicMock()
        scheme_divisor = MagicMock()

        dividend = MagicMock(
            expand=MagicMock(return_value=scheme_dividend)
        )
        divisor = MagicMock(
            expand=MagicMock(return_value=scheme_divisor)
        )

        op = LazyOperation.division(dividend, divisor)
        op.expand()

        scheme_dividend.__truediv__.assert_called_once()
        scheme_divisor.__truediv__.assert_not_called()


class OperatorTest(unittest.TestCase):

    def test_Expand_NoElement_ReturnStencilSchemeShiftedToNodeAddress(self):

        stencil_weights = {-1: 1., 1: 1.}
        stencil = Stencil(stencil_weights)
        operator = Operator(stencil, element=None)

        result = operator.expand(-2)

        expected = Scheme(stencil_weights).shift(-2)

        self.assertEqual(expected, result)

    def _build_operator(self, *data):
        return Operator(
            {i: d for i, d in enumerate(data)}
        )


class NumberTest(unittest.TestCase):
    def test_Expand_Float_ReturnFloat(self):

        value = 3.
        number = Number(value)

        result = number.expand(1)

        expected = value

        self.assertEquals(expected, result)

    def test_Expand_Callable_ReturnValueComputedByCallable(self):

        value = 999.
        number = Number(lambda node_address: value)

        result = number.expand(1)

        expected = value

        self.assertEquals(expected, result)


class NodeFunctionTest(unittest.TestCase):
    def test_Get_IntegerNodeAddress_ReturnValueForNode(self):

        _node_address = 2.

        def value_in_node(node):
            return node

        _function = NodeFunction(value_in_node)

        result = _function.get(_node_address)

        expected = _node_address

        self.assertEqual(expected, result)

    @patch('fractulus.logger.solver')
    def test_Get_FloatNodeAddressCloserToLeftNode_ReturnValueForLeftNode(self, solver_logger):

        def value_in_node(node):
            return node

        _function = NodeFunction(value_in_node)

        result = _function.get(2.2)

        expected = 2

        self.assertEqual(expected, result)

    @patch('fractulus.logger.solver')
    def test_Get_FloatNodeAddressCloserToRightNode_ReturnValueForRightNode(self, solver_logger):
        def value_in_node(node):
            return node

        _function = NodeFunction(value_in_node)

        result = _function.get(2.6)

        expected = 3

        self.assertEqual(expected, result)

    def test_Get_FloatNodeAddressAndInterpolator_ReturnInterpolatedValue(self):

        def value_in_node(node):
            return node

        def interpolator(x, x1, x2, v1, v2):
            return v1 + (v2 - v1)*(x - x1)/(x2 - x1)

        _function = NodeFunction(value_in_node, interpolator)

        result = _function.get(2.2)

        expected = interpolator(0.2, 1., 2., 1., 2.)

        self.assertAlmostEqual(expected, result)