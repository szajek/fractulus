import unittest
import numpy as np
from mock import MagicMock

from fractulus.finite_difference import Scheme, LinearEquationTemplate
from fractulus.system import LinearEquation,  model_to_equations, VirtualNode, EquationWriter, VirtualNodeWriter, \
    extract_virtual_nodes, Output, VirtualValueStrategy


def create_domain(node_number, delta=2.):
    domain = MagicMock(
        nodes=[MagicMock() for i in range(node_number)],
        calc_average_delta=lambda i: delta,
    )
    return domain


class ExtractVirtualNodesTest(unittest.TestCase):
    def test_Call_ExistAndSymmetryStrategy_ReturnVirtualNode(self):
        domain = create_domain(3)
        eq = LinearEquation({-1: 2., 3: 3., 1: 1.}, 1.)

        self.assertEqual(
            [VirtualNode(3, 1), VirtualNode(-1, 1)],
            extract_virtual_nodes(eq, domain, strategy=VirtualValueStrategy.SYMMETRY)
        )

    def test_Call_ExistAndAsBorderStrategy_ReturnVirtualNode(self):
        domain = create_domain(3)
        eq = LinearEquation({-1: 2., 3: 3., 1: 1.}, 1.)

        self.assertEqual(
            [VirtualNode(3, 2), VirtualNode(-1, 0)],
            extract_virtual_nodes(eq, domain, strategy=VirtualValueStrategy.AS_IN_BORDER)
        )


class ModelToEquationsTest(unittest.TestCase):
    def test_Call_Always_ReturnEquationsCreatedBasedOnGivenTemplateAndDomain(self):

        coefficients = lambda i: Scheme({i: 1})
        free_value = lambda i: i

        equation = LinearEquationTemplate(coefficients, free_value)

        model = MagicMock(
            domain=MagicMock(
                nodes=[
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ],
                get_connections=MagicMock(
                    return_value=[
                    MagicMock(length=2.),
                    MagicMock(length=2.),
                ]
                )
            ),
            bcs=[],
            equation=equation,
        )

        equations = model_to_equations(model)

        self.assertEqual({0: 1/2.}, equations[0].coefficients)
        self.assertEqual(0., equations[0].free_value)


class EquationWriterTest(unittest.TestCase):

    def test_ToWeightArray_RenumeratorNotProvided_ReturnArrayWithWeights(self):

        eq = LinearEquation({0: 2.}, 1.)

        writer = EquationWriter(eq, {})

        result = writer.to_coefficients_array(3)

        np.testing.assert_allclose(np.array(
            [2., 0., 0.],
        ), result)

    def test_ToWeightArray_RenumeratorProvided_ReturnArrayWithWeightsInRenumberedPosition(self):

        eq = LinearEquation({0: 2.}, 1.)

        writer = EquationWriter(eq, {0: 1})

        result = writer.to_coefficients_array(3)

        np.testing.assert_allclose(np.array(
            [0., 2., 0.],
        ), result)

    def test_ToFreeValue_Always_ReturnFreeValue(self):

        eq = LinearEquation({0: 2.}, 1.)

        writer = EquationWriter(eq, {0: 1})

        result = writer.to_free_value()

        np.testing.assert_allclose(np.array(1.,), result)


class VirtualNodeWriterTest(unittest.TestCase):
    def test_ToWeightArray_Always_ReturnArrayWithWeights_SymmetryConsidered(self):
        eq = VirtualNode(-1, 1)

        writer = VirtualNodeWriter(eq, 0, 2)

        result = writer.to_coefficients_array(3)

        np.testing.assert_allclose(np.array(
            [0., -1., 1.],
        ), result)


class OutputTest(unittest.TestCase):
    def test_GetItem_IndexInDomain_ReturnValueInRealNode(self):
        o = Output([1, 2, 3], 2, {})
        self.assertEquals(
            2.,
            o[1]
        )

    def test_GetItem_NegativeIndex_ReturnValueInVirtualNode(self):
        address_forwarder = {-1: 2}
        o = Output([1, 2, 3, 4], 2, address_forwarder)
        self.assertEquals(
            3.,
            o[-1]
        )

    def test_GetItem_NegativeIndex_ReturnValueInVirtualNode(self):
        address_forwarder = {2: 2}
        o = Output([1, 2, 3], 2, address_forwarder)
        self.assertEquals(
            3.,
            o[2]
        )