import collections

from finite_difference import Stencil


__all__ = ['Model', 'BoundaryCondition']


Model = collections.namedtuple("Model", ('equation', 'domain', 'bcs'))

_BoundaryConditions = collections.namedtuple('_BoundaryConditions', ('coefficients', 'free_value'))


class BoundaryCondition(_BoundaryConditions):
    @classmethod
    def dirichlet(cls, value=0.):
        return cls(Stencil({0: 1.}), lambda *args: value)

    @classmethod
    def neumann(cls, stencil):
        return cls(stencil, lambda *args: 0.)
