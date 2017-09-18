import unittest
import numpy as np

from fdm.domain import Grid1DBuilder
from fdm.equation import Operator, Stencil, Number, LinearEquationTemplate, NodeFunction
from fdm.model import BoundaryCondition, Model
from fdm.system import solve
from fractulus.equation import CaputoSettings, create_fractional_deformation_operator


def _create_linear_function(length, node_number, a, b):
    def calc(node_address):
        x = (node_address / (node_number - 1) * length)
        return a*x + b
    return calc


def _create_domain(length, node_number):
    domain_builder = Grid1DBuilder(length)
    domain_builder.add_uniformly_distributed_nodes(node_number)
    return domain_builder.create()


def _create_equation(linear_operator, free_vector):
    return LinearEquationTemplate(
        linear_operator,
        free_vector
    )


def _build_fractional_operator(E, A, settings):
    return Operator(
        Stencil.central(1.),
        Number(A) * Number(E) * create_fractional_deformation_operator(settings)
    )


def _create_fixed_and_free_end_bc(node_number):
    return {
        0: BoundaryCondition.dirichlet(),
        node_number - 1: BoundaryCondition.neumann(Stencil.backward())
    }


def _create_fixed_ends_bc(node_number):
    return {
        0: BoundaryCondition.dirichlet(),
        node_number - 1: BoundaryCondition.dirichlet()
    }


_bcs = {
    'fixed_free': _create_fixed_and_free_end_bc,
    'fixed_fixed': _create_fixed_ends_bc,
}


def _create_bc(_type, node_number):
    return _bcs[_type](node_number)


class TrussStaticEquationFractionalDifferencesTest(unittest.TestCase):
    def test_ConstantSection_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.99999, .5, 5)

        results = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_free', settings, load_function_coefficients=(-1., 0.))

        expected = np.array(
                [
                    [0.],
                    [0.08],
                    [0.152],
                    [0.208],
                    [0.24],
                    [0.24],
                ]
            )

        np.testing.assert_allclose(expected, results, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha05_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.5, 3, 3)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_fixed', settings, load_function_coefficients=(0., -1.))

        expected = np.array(
            [[9.41385462e-16],
             [3.47172214e-01],
             [4.99518990e-01],
             [4.99518990e-01],
             [3.47172214e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_Alpha03_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.3, 3, 3)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_fixed', settings, load_function_coefficients=(0., -1.))

        expected = np.array(
            [[5.46428567e-16],
             [9.59393722e-01],
             [1.74524563e+00],
             [1.74524563e+00],
             [9.59393722e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_AlphaAlmostOne_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.9999, 3, 3)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_fixed', settings, load_function_coefficients=(0., -1.))

        expected = np.array(
            [[8.88464753e-17],
             [8.00209890e-02],
             [1.20028113e-01],
             [1.20028113e-01],
             [8.00209890e-02],
             [0.00000000e+00], ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlpha05_ReturnCorrectDisplacement(self):

        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.5, 2.5, 6)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_fixed', settings, load_function_coefficients=(0., -1.))

        expected = np.array(
            [[-2.79921788e-16],
             [2.93429923e-01],
             [3.71316963e-01],
             [3.71316963e-01],
             [2.93429923e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlpha03_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.3, 2.5, 6)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_fixed', settings, load_function_coefficients=(0., -1.))

        expected = np.array(
            [[1.42549063e-16],
             [5.85829500e-01],
             [7.06500004e-01],
             [7.06500004e-01],
             [5.85829500e-01],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_ConstantSectionFixedEnds_LfDifferentThanResolutionAndAlphaAlmostOne_ReturnCorrectDisplacement(self):

        domain = _create_domain(length=1., node_number=6)
        settings = CaputoSettings(0.9999, 2.5, 6)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_fixed', settings, load_function_coefficients=(0., -1.))

        expected = np.array(
            [[7.10756467e-17],
             [8.00189611e-02],
             [1.20024393e-01],
             [1.20024393e-01],
             [8.00189611e-02],
             [0.00000000e+00]]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def test_VariedSection_Always_ReturnCorrectDisplacement(self):

        length, node_number = 1., 6
        domain = _create_domain(length, node_number)
        settings = CaputoSettings(0.9999, .5, 5)

        result = _solve_for_fractional('linear_system_of_equations', domain, 'fixed_free', settings, load_function_coefficients=(0., -1.),
                                       cross_section=NodeFunction.with_linear_interpolator(
                                           _create_linear_function(length, node_number, -1. / length, 2.)
                                       ))

        expected = np.array(
            [[-3.92668354e-16],
             [8.42105263e-02],
             [1.54798762e-01],
             [2.08132095e-01],
             [2.38901326e-01],
             [2.38901326e-01],
             ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)


def _solve_for_fractional(analysis_type, domain, bc_type, settings, load_function_coefficients, cross_section=1.):
    a, b = load_function_coefficients
    node_number = len(domain.nodes)
    length = domain.boundary_box.dimensions[0]
    return solve(
        analysis_type,
        Model(
            _create_equation(
                _build_fractional_operator(A=cross_section, E=1, settings=settings),
                _create_linear_function(length, node_number, a=a, b=b),
            ),
            domain,
            _create_bc(bc_type, node_number))
    )


class TrussDynamicEigenproblemEquationFractionalDifferencesTest(TrussStaticEquationFractionalDifferencesTest):
    @unittest.skip("No result to compare")
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):  # todo: compute result to compare
        domain = _create_domain(length=1., node_number=101)
        settings = CaputoSettings(0.8, 10, 10)
        ro = 2.

        result = _solve_for_fractional('eigenproblem', domain, 'fixed_fixed', settings,
                                       load_function_coefficients=(0., -ro))

        expected = np.array(
            [0.,],  # no result to compare
        )

        np.testing.assert_allclose(expected, result, atol=1e-4)
