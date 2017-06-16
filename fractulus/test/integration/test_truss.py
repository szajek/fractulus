import sys
import unittest

import numpy as np
from fractulus.domain import Grid1DBuilder
from fractulus.finite_difference import Operator, Stencil, Number, LinearEquationTemplate, NodeFunction
from fractulus.fractional_difference import CaputoSettings, create_fractional_deformation_operator
from fractulus.model import BoundaryCondition, Model
from fractulus.system import solve


def _create_domain(length, node_number):
    domain_builder = Grid1DBuilder(length)
    domain_builder.nodes_by_number(node_number)
    domain = domain_builder.create()
    return domain


class TrussStaticEquationFiniteDifferencesTest(unittest.TestCase):
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):
        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        u = solve(
            'linear_system_of_equations',
            Model(
                self._create_equation(
                    self._create_standard_operator(A=1., E=1.),
                    lambda node_address: -node_address / last_node_idx * length
                ),
                self._create_domain(length, node_number),
                self._create_fixed_and_free_end_bc(last_node_idx)))

        np.testing.assert_allclose(np.array(
            [
                [0.],
                [0.08],
                [0.152],
                [0.208],
                [0.24],
                [0.24],
            ]
        ), u, atol=1e-6)

    def test_VariedSection_ReturnCorrectDisplacement(self):
        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        def A(node_address):
            x = domain.get_by_address(node_address).x
            return 2. - (x / length) * 1.

        u = solve(
            'linear_system_of_equations',
            Model(
            self._create_equation(
                self._create_standard_operator(A=NodeFunction.with_linear_interpolator(A), E=1.),
                lambda node_address: -1.
            ),
            domain,
            self._create_fixed_and_free_end_bc(last_node_idx)
        ))

        np.testing.assert_allclose(np.array(
            [[-3.92668354e-16],
             [8.42105263e-02],
             [1.54798762e-01],
             [2.08132095e-01],
             [2.38901326e-01],
             [2.38901326e-01],
             ]
        ), u, atol=1e-6)

    def _create_domain(self, length, node_number):
        domain_builder = Grid1DBuilder(length)
        domain_builder.nodes_by_number(node_number)
        domain = domain_builder.create()
        return domain

    def _create_equation(self, linear_operator, free_vector):
        return LinearEquationTemplate(
            linear_operator,
            free_vector
            )

    def _create_fixed_and_free_end_bc(self, last_node_idx):
        return {
            0: BoundaryCondition.dirichlet(),
            last_node_idx: BoundaryCondition.neumann(Stencil.backward())
        }

    def _create_fixed_ends_bc(self, last_node_idx):
        return {
            0: BoundaryCondition.dirichlet(),
            last_node_idx: BoundaryCondition.dirichlet()
        }

    def _create_standard_operator(self, A, E):
        ep_central = Operator(
            Stencil.central(1.),
            Number(A) * Number(E) * Operator(
                Stencil.central(1.),
            )
        )
        return ep_central


class TrussStaticEquationFractionalDifferencesTest(TrussStaticEquationFiniteDifferencesTest):

    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):

        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        u = solve(
            'linear_system_of_equations',
            Model(
                self._create_equation(
                    self._build_fractional_operator(A=1., E=1, settings=CaputoSettings(0.99999, .5, 5)),
                    lambda node_address: -node_address / last_node_idx * length,
                ),
                domain,
                self._create_fixed_and_free_end_bc(last_node_idx))
        )

        np.testing.assert_allclose(np.array(
            [
                [0.],
                [0.08],
                [0.152],
                [0.208],
                [0.24],
                [0.24],
            ]
        ), u, atol=1e-4)

    def test_ConstantSectionAndYoungFixedEnds_IntegrateNodeAgreeWithDomainNodes_ReturnCorrectDisplacement(self):

        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        def _solve(settings):
            return solve(
                'linear_system_of_equations',
                Model(
                    self._create_equation(
                        self._build_fractional_operator(A=1., E=1, settings=settings),
                        lambda node_address: -1.,
                    ),
                    domain,
                    self._create_fixed_ends_bc(last_node_idx),
                )
            )

        u1 = _solve(CaputoSettings(0.5, 3, 3))
        u3 = _solve(CaputoSettings(0.3, 3, 3))
        u2 = _solve(CaputoSettings(0.9999, 3, 3))

        if '--draw-graph' in sys.argv:
            from graph import draw
            draw([domain, domain, domain], [u1, u2, u3])

        np.testing.assert_allclose(np.array(
            [[9.41385462e-16],
             [3.47172214e-01],
             [4.99518990e-01],
             [4.99518990e-01],
             [3.47172214e-01],
             [0.00000000e+00]]
        ), u1, atol=1e-4)

        np.testing.assert_allclose(np.array(
            [[8.88464753e-17],
             [8.00209890e-02],
             [1.20028113e-01],
             [1.20028113e-01],
             [8.00209890e-02],
             [0.00000000e+00],]
        ), u2, atol=1e-4)

        np.testing.assert_allclose(np.array(
            [[5.46428567e-16],
             [9.59393722e-01],
             [1.74524563e+00],
             [1.74524563e+00],
             [9.59393722e-01],
             [0.00000000e+00]]
        ), u3, atol=1e-4)

    def test_ConstantSectionAndYoungFixedEnds_IntegrateNodeNOTAgreeWithDomainNodes_ReturnCorrectDisplacement(self):

        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        bcs = {
            0: BoundaryCondition.dirichlet(),
            last_node_idx: BoundaryCondition.dirichlet()
        }

        def _solve(settings):
            return solve(
                'linear_system_of_equations',
                Model(
                    self._create_equation(
                        self._build_fractional_operator(A=1., E=1, settings=settings),
                        lambda node_address: -1.,
                    ),
                    domain,
                    bcs,
                )
            )

        u1 = _solve(CaputoSettings(0.5, 2.5, 6))
        u3 = _solve(CaputoSettings(0.3, 2.5, 6))
        u2 = _solve(CaputoSettings(0.9999, 2.5, 6))

        if '--draw-graph' in sys.argv:
            from graph import draw
            draw([domain, domain, domain], [u1, u2, u3])

        np.testing.assert_allclose(np.array(
            [[-2.79921788e-16],
             [2.93429923e-01],
             [3.71316963e-01],
             [3.71316963e-01],
             [2.93429923e-01],
             [0.00000000e+00]]
        ), u1, atol=1e-4)

        np.testing.assert_allclose(np.array(
            [[7.10756467e-17],
             [8.00189611e-02],
             [1.20024393e-01],
             [1.20024393e-01],
             [8.00189611e-02],
             [0.00000000e+00]]
        ), u2, atol=1e-4)

        np.testing.assert_allclose(np.array(
            [[1.42549063e-16],
             [5.85829500e-01],
             [7.06500004e-01],
             [7.06500004e-01],
             [5.85829500e-01],
             [0.00000000e+00]]
        ), u3, atol=1e-4)

    def test_FractionalDifferences_VariedSection_ReturnCorrectDisplacement(self):

        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        def A(node_address):
            x = domain.get_by_address(node_address).x
            return 2. - (x / length) * 1.

        u = solve(
            'linear_system_of_equations',
            Model(
                self._create_equation(
                    self._build_fractional_operator(A=NodeFunction.with_linear_interpolator(A), E=1.,
                                                    settings=CaputoSettings(0.9999, .5, 5)),
                    lambda node_address: -1.
                ),
                domain,
                self._create_fixed_and_free_end_bc(last_node_idx)
            ))

        np.testing.assert_allclose(np.array(
            [[-3.92668354e-16],
             [8.42105263e-02],
             [1.54798762e-01],
             [2.08132095e-01],
             [2.38901326e-01],
             [2.38901326e-01],
             ]
        ), u, atol=1e-4)

    def _build_fractional_operator(self, E, A, settings):
        ep_central = Operator(
            Stencil.central(1.),
            Number(A) * Number(E) * create_fractional_deformation_operator(settings)
        )
        return ep_central


class TrussDynamicEigenproblemEquationFiniteDifferencesTest(TrussStaticEquationFiniteDifferencesTest):
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):
        node_number, length = 6, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        ro = 2.

        bcs = {
            0: BoundaryCondition.dirichlet(),
            last_node_idx: BoundaryCondition.dirichlet(),
        }

        u = solve(
            'eigenproblem',
            Model(
                self._create_equation(
                    self._create_standard_operator(A=1., E=1.),
                    lambda node_address: -ro
                ),
                domain,
                bcs),
        )

        np.testing.assert_allclose(np.array(
                [0., -0.3717, -0.6015, -0.6015, -0.3717, 0.],
        ), u, atol=1e-4)

    def _test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):  # todo: not verified yet
        node_number, length = 50, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        ro = 2.

        lf = 3.0

        bcs = {
            0: BoundaryCondition.dirichlet(),
            last_node_idx: BoundaryCondition.dirichlet(),
        }

        frac_ep_central = self._build_fractional_operator(A=1., E=1., settings=CaputoSettings(0.8, lf, 3))
        class_ep_central = self._create_standard_operator(A=1., E=1.)

        def frac_weights(address):
            offset = lf
            if address > offset and address < last_node_idx - offset:
                return frac_ep_central(address)
            else:
                return class_ep_central(address)

        ep_central = frac_weights
        eq = LinearEquationTemplate(
            ep_central,
            lambda node_address: -ro
        )

        u = solve(
            'eigenproblem',
            Model(
                eq,
                domain,
                bcs),
        )


class TrussDynamicEigenproblemEquationFractionalDifferencesTest(TrussStaticEquationFractionalDifferencesTest):
    def _test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):
        node_number, length = 101, 1.
        last_node_idx = node_number - 1

        domain = self._create_domain(length, node_number)

        ro = 2.

        bcs = {
            0: BoundaryCondition.dirichlet(),
            last_node_idx: BoundaryCondition.dirichlet(),
        }

        u = solve(
            Model(
                self._create_equation(
                    self._build_fractional_operator(A=1., E=1., settings=CaputoSettings(0.8, 10, 10)),
                    lambda node_address: -ro
                ),
                domain,
                bcs),
            eig=True
        )

        from graph import draw
        draw([domain], [u])

        np.testing.assert_allclose(np.array(
                [0., -0.3717, -0.6015, -0.6015, -0.3717, 0.],
        ), u, atol=1e-4)