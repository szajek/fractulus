import math
import unittest
from mock import MagicMock, patch

from fractulus.finite_difference import (LazyOperation, Operator, Stencil, Scheme, Number, Element, \
    Delta, NodeFunction, operate, merge_weights, Coefficients, MutateMixin)


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

    def test_Mutate_PublicField_ReturnNewObjectWithProvidedOrObjectFieldsValues(self):
        obj = self.PublicFieldsExample('field_1_value', 'field_2_value')

        mutated = obj.mutate(field_1='new')

        self.assertEqual(
            'new',
            mutated.field_1
        )

        self.assertEqual(
            'field_2_value',
            mutated.field_2
        )

    def test_Mutate_PrivateField_ReturnNewObjectWithProvidedOrObjectFieldsValues(self):
        obj = self.PrivateFieldsExample('field_1_value', 'field_2_value')

        mutated = obj.mutate(field_1='new')

        self.assertEqual(
            'new',
            mutated._field_1
        )

        self.assertEqual(
            'field_2_value',
            mutated._field_2
        )


class DeltaTest(unittest.TestCase):
    def test_Create_Always_CalculateAverageValueForGivenDeltas(self):
        d = Delta(1., 2.)

        self.assertEquals(1.5, d.average)

    def test_MathOperations_Always_UseAverageValueAndReturnFloat(self):
        d = Delta(1., 2.)
        avg = d.average

        self.assertEqual(avg + 1., d + 1.)
        self.assertEqual(avg * 2., d * 2.)
        self.assertEqual(avg - 2., d - 2.)
        self.assertEqual(avg / 2., d / 2.)


class CoefficientsTest(unittest.TestCase):

    def test_ToValue_OutputProvided_ReturnNumberAsMultiplicationOfCoefficientsAndVariableValues(self):
        self.assertEqual(
            1.3 * 2.1 + 2.3 * 1.1,
            Coefficients({1: 1.3, 2: 2.3}).to_value({1: 2.1, 2: 1.1})
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

    def test_Duplicate_Always_ReturnCopyOfScheme(self):
        s1 = Scheme({1: 2, 3: 1.})
        self.assertEqual(
            s1,
            s1.duplicate()
        )
        self.assertNotEqual(
            id(s1),
            id(s1.duplicate())
        )

    def test_Add_None_ReturnCurrentScheme(self):
        s1 = Scheme({1: 2, 3: 1.})
        s2 = None
        self.assertEqual(
            s1,
            s1 + s2
        )

    def test_RightAdd_None_ReturnCurrentScheme(self):
        s1 = None
        s2 = Scheme({1: 2, 3: 1.})
        self.assertEqual(
            s2,
            s1 + s2
        )

    def test_Add_DisjoinedSetsOfNodeIndices_MergeData(self):
        s1 = Scheme({1: 2})
        s2 = Scheme({2: 3})
        self.assertEqual(
            Scheme({1: 2, 2: 3}),
            s1 + s2
        )

    def test_Add_NodeIndicesSetsWithIntersection_MergeForUniqueAndAddForIntersection(self):
        s1 = Scheme({1: 2, 3: 1.})
        s2 = Scheme({2: 3, 3: 2.})
        self.assertEqual(
            Scheme({1: 2, 2: 3, 3: 3}),
            s1 + s2
        )

    def test_Add_InconsistentOrder_ThrowsAttributeException(self):

        s1 = Scheme({}, order=1)
        s2 = Scheme({}, order=2)

        with self.assertRaises(AttributeError):
            s1 + s2

    def test_Shift_Number_ShiftNodeAddressesByGivenNumber(self):
        self.assertEquals(
            Scheme({-1: 1., 2.: -3}),
            Scheme({0: 1., 3.: -3}).shift(-1.)
        )

    def test_LeftMultiplication_IntegerOrFloat_MultDataElementsByGivenValue(self):
        self.assertEqual(self._build_scheme((0., 4., 6.)), 2 * self._build_scheme((0., 2., 3.)))

    def test_RightMultiplication_IntegerOrFloat_MultDataElementsByGivenValue(self):
        self.assertEqual(self._build_scheme((0., 4., 6.)), self._build_scheme((0., 2., 3.)) * 2.)

    def test_ToCoefficients_Always_ReturnCoeffsDictConsideringGivenDeltaAndSchemeOrder(self):

        self.assertEquals({1: 0.5}, Scheme({1: 1}, 1).to_coefficients(2.))
        self.assertEquals({1: 0.25}, Scheme({1: 1}, 2).to_coefficients(2.))

    def test_ToCoefficients_WeightsBetweenNodes_SpreadWeightsAroundTheClosestNodes(self):
        self.assertEquals({0: 0.5, 1: 0.5}, Scheme({0.5: 1}).to_coefficients(1.))
        self.assertEquals({0: 0.5, -1: 0.5}, Scheme({-0.5: 1}).to_coefficients(1.))
        self.assertEquals({0: 0.75, -1: 0.25}, Scheme({-0.25: 1.}).to_coefficients(1.))
        self.assertEquals({0: 0.25, 1: 0.75}, Scheme({0.75: 1}).to_coefficients(1.))

    def _build_scheme(self, data, order=1.):
        return Scheme(
            {i: d for i, d in enumerate(data)},
            order
        )


class MergeWeightsTest(unittest.TestCase):
    def test_Call_Disjoined_ReturnDictWithElementsFromBoth(self):
        self.assertEqual(merge_weights({1: 1}, {2: 2}), {1: 1, 2: 2})

    def test_Call_SharedNodesAddresses_ReturnDictWithElementsFromBothAndSummedWeightsForSharedAddresses(self):
        self.assertEqual(merge_weights({1: 1, 3: 1}, {2: 2, 3: 2}), {1: 1, 2: 2, 3: 3})

    def test_Call_ManyGiven_ReturnMergedElementsFromAllGiven(self):
        self.assertEqual(merge_weights({1: 1}, {2: 2}, {3: 3}), {1: 1, 2: 2, 3: 3})


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

    def test_Call_WithScheme_ReturnSchemeWithWeightsReplacesByGivenScheme(self):
        scheme = Scheme({1: 2})

        element = MagicMock(
            expand=lambda node_idx: Scheme({0: 1., 5.: 2.})
        )

        result = operate(scheme, element)

        self.assertEqual(Scheme({0: 1. * 2., 5.: 2. * 2.}, order=2.), result)

    def test_Call_WithFractionalScheme_ReturnSchemeWithSummedOrders(self):
        scheme = Scheme({1: 1}, order=1.2)

        element = Stencil({1: 1}, order=2.2)

        scheme = operate(scheme, element)

        self.assertAlmostEqual(3.4, scheme.order)

    def test_Call_WithSchemeVariedByNodeAddress_ReturnRevolvedScheme(self):
        scheme = Scheme({1: 2, 2: 4})

        element = MagicMock(
            expand=lambda node_idx: {
                1: Scheme({0: 1., 6.: 3.}),
                2: Scheme({0: 1., 5.: 2.}),
            }[node_idx]
        )

        result = operate(scheme, element)

        self.assertEqual(Scheme({0: 1.*2 + 1.*4., 5.: 2.*4., 6: 3. * 2.}, order=2.), result)

    def test_Call_WithNumber_ReturnSchemeOfSchemeOrder(self):

        scheme_order = 1.2
        scheme = Scheme({1: 1}, order=scheme_order)

        element = Number(0)

        scheme = operate(scheme, element)

        self.assertAlmostEqual(scheme_order, scheme.order)

#


class ElementTest(unittest.TestCase):
    class ConcreteElement(Element):
        def expand(self):
            return Scheme({1: 1})

    def test_Adding_TwoElements_ReturnLazyOperationWithElements(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 + e2

        self.assertEquals(
            LazyOperation.summation(e1, e2),
            result
        )

    def test_Multiplicatin_TwoElements_ReturnLazyOperationWithElements(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 * e2

        self.assertEquals(
            LazyOperation.multiplication(e1, e2),
            result
        )

    def test_Subtraction_TwoElements_ReturnLazyOperationWithElements(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 - e2

        self.assertEquals(
            LazyOperation.subtraction(e1, e2),
            result
        )

    def test_Division_TwoElements_ReturnLazyOperationWithElements(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 / e2

        self.assertEquals(
            LazyOperation.division(e1, e2),
            result
        )


class StencilTest(unittest.TestCase):
    def test_Create_NoAxesProvided_AssignAxisOneByDefault(self):
        a = Stencil(1.)
        self.assertEqual(
            1,
            a._axis
        )

    def test_Expand_OrderDifferentThanOne_CreateSchemeWithGivenOrder(self):

        s = Stencil({}, order=1.2)
        self.assertEqual(
            1.2,
            s.expand(0.)._order,
        )

    def test_Uniform_Always_GenerateUniformlyDistributedNodesInGivenLimits(self):
        left_range, right_range = 1., 2.
        resolution = 3.

        def weights_provider(i, stencil_node_address):
            return stencil_node_address

        s = Stencil.uniform(left_range, right_range, resolution, weights_provider)

        _range = right_range + left_range
        delta = _range/resolution
        left_limit = -left_range
        expected_node_1 = left_limit
        expected_node_2 = left_limit + delta * 1.
        expected_node_3 = left_limit + delta * 2.
        expected_node_4 = left_limit + delta * 3.

        self.assertEquals(
            {
                expected_node_1: expected_node_1,
                expected_node_2: expected_node_2,
                expected_node_3: expected_node_3,
                expected_node_4: expected_node_4,
            },
            s._weights
        )

    def test_Central_RangeOne_GenerateWeightsForMidnodes(self):
        self.assertTrue(self._compare_dict(
            Stencil({-0.5: -1., 0.5: 1.})._weights,
            Stencil.central(1.)._weights,
        ))

    def test_Central_RangeTwo_GenerateWeightsInNodeDividedByTwo(self):
        self.assertTrue(self._compare_dict(
            Stencil({-1.: -0.5, 1.: 0.5})._weights,
            Stencil.central(2.)._weights,
        ))
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

    def test_Expand_SingleOperator_ReturnCoeffsWithNodesNumbersMovedByGivenValue(self):
        s = Operator(Stencil.central())
        self.assertEqual(Scheme({-3: -0.5, -1: 0.5}), s.expand(-2))

    def _build_operator(self, *data):
        return Operator(
            {i: d for i, d in enumerate(data)}
        )


class NumberTest(unittest.TestCase):
    def test_Expand_Float_ReturnFloat(self):
        self.assertEquals(3, Number(3).expand(1))

    def test_Expand_Callable_ReturnGivenValue(self):
        _callable = lambda node_address: node_address * 2
        self.assertEquals(1. * 2., Number(_callable).expand(1))


class NodeFunctionTest(unittest.TestCase):
    def test_Get_ExactNode_ReturnValueForNode(self):
        def value_in_node(node):
            return node

        nf = NodeFunction(value_in_node)

        self.assertEqual(
            2.,
            nf.get(2.)
        )

    def test_Get_BetweenNodesInterpolatorProvided_ReturnInterpolatedValue(self):
        def value_in_node(node):
            return 2.**node

        def interpolator(x, x1, x2, v1, v2):
            return 3.33

        nf = NodeFunction(value_in_node, interpolator)

        self.assertEqual(
            3.33,
            nf.get(2.2)
        )

    @patch('fractulus.logger.solver')
    def test_Get_BetweenNodesInterpolatorNodProvided_ReturnValueForClosestNode(self, solver_logger):
        def value_in_node(node):
            return 2.**node

        nf = NodeFunction(value_in_node)

        self.assertEqual(
            2**2,
            nf.get(2.2)
        )
        self.assertEqual(
            2**3,
            nf.get(2.6)
        )