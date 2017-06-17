import unittest

import numpy as np
from mock import MagicMock

from fractulus.finite_difference import Scheme, LinearEquationTemplate
from fractulus.system import (LinearEquation, model_to_equations, VirtualNode, EquationWriter, VirtualNodeWriter,
                              extract_virtual_nodes, Output, VirtualValueStrategy)


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

        result = extract_virtual_nodes(eq, domain, strategy=VirtualValueStrategy.SYMMETRY)

        expected = [VirtualNode(3, 1), VirtualNode(-1, 1)]

        self.assertEqual(expected, result)

    def test_Call_ExistAndAsBorderStrategy_ReturnVirtualNode(self):
        domain = create_domain(3)
        eq = LinearEquation({-1: 2., 3: 3., 1: 1.}, 1.)

        result = extract_virtual_nodes(eq, domain, strategy=VirtualValueStrategy.AS_IN_BORDER)

        expected = [VirtualNode(3, 2), VirtualNode(-1, 0)]

        self.assertEqual(expected, result)


class ModelToEquationsTest(unittest.TestCase):
    def test_Call_Always_ReturnEquationsCreatedBasedOnGivenTemplateAndDomain(self):
        def get_scheme(i):
            return Scheme({i: i})

        def get_free_value(i):
            return i

        node_span = 2.
        node_number = 3

        equation = LinearEquationTemplate(get_scheme, get_free_value)

        model = MagicMock(
            domain=MagicMock(
                nodes=[MagicMock() for node_address in range(node_number)],
                get_connections=MagicMock(
                    return_value=[MagicMock(length=node_span) for node_address in range(node_number - 1)]
                )
            ),
            bcs=[],
            equation=equation,
        )

        equations = model_to_equations(model)

        for equation_number in range(0, 3):
            equation = equations[equation_number]

            expected_coefficients = {equation_number: equation_number/node_span}
            expected_free_value = get_free_value(equation_number)

            self.assertEqual(expected_coefficients, equation.coefficients)
            self.assertEqual(expected_free_value, equation.free_value)


class EquationWriterTest(unittest.TestCase):
    def test_ToCoefficientsArray_RenumeratorNotProvided_ReturnArrayWithWeights(self):
        eq = LinearEquation({0: 2.}, 1.)
        writer = EquationWriter(eq, {})

        result = writer.to_coefficients_array(3)

        expected = np.array([2., 0., 0.])

        np.testing.assert_allclose(expected, result)

    def test_ToCoefficientsArray_RenumeratorProvided_ReturnArrayWithWeightsInRenumberedPosition(self):
        eq = LinearEquation({0: 2.}, 1.)
        writer = EquationWriter(eq, {0: 1})

        result = writer.to_coefficients_array(3)

        expected = np.array([0., 2., 0.],)

        np.testing.assert_allclose(expected, result)

    def test_ToFreeValue_Always_ReturnFreeValue(self):
        eq = LinearEquation({0: 2.}, 1.)
        writer = EquationWriter(eq, {0: 1})

        result = writer.to_free_value()

        expected = np.array(1., )

        np.testing.assert_allclose(expected, result)


class VirtualNodeWriterTest(unittest.TestCase):
    def test_ToCoefficientsArray_Always_ReturnArrayWithWeights_SymmetryConsidered(self):
        eq = VirtualNode(-1, 1)
        writer = VirtualNodeWriter(eq, 0, 2)

        result = writer.to_coefficients_array(3)

        expected = np.array([0., -1., 1.],)

        np.testing.assert_allclose(expected, result)


class OutputTest(unittest.TestCase):
    def test_GetItem_IndexInDomain_ReturnValueInRealNode(self):
        value = 2
        o = Output([1, value, 3], 2, {})

        result = o[1]

        expected = value

        self.assertEquals(expected, result)

    def test_GetItem_NegativeIndex_ReturnValueInVirtualNode(self):
        value = 3.
        virtual_node_address = -1
        address_forwarder = {virtual_node_address: 2}
        o = Output([1, 2, value, 4], 2, address_forwarder)

        result = o[virtual_node_address]

        expected = value

        self.assertEquals(expected, result)

    def test_GetItem_PositiveIndex_ReturnValueInVirtualNode(self):
        value = 2
        virtual_node_address = 3
        address_forwarder = {virtual_node_address: 2}
        o = Output([1, 2, value], 2, address_forwarder)

        result = o[virtual_node_address]

        expected = value

        self.assertEquals(expected, result)
