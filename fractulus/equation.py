import collections
import math

from fdm.equation import Stencil, Number
from fdm.geometry import Point

__all__ = ['CaputoSettings', 'create_left_caputo_stencil', 'create_right_caputo_stencil', 'create_riesz_caputo_stencil']


CaputoSettings = collections.namedtuple('CaputoSettings', ('alpha', 'lf', 'resolution'))


def create_side_caputo_stencil(alpha, p, start, end, left_weight_provider, right_weight_provider,
                               interior_weights_provider, multiplier):
    n = math.floor(alpha) + 1.
    index = (n - alpha + 1.)
    _range = end - start
    h = _range/p
    _multiplier = multiplier(n) * ((h**(n - alpha)) / math.gamma(n - alpha + 2.))

    def pure_weight_provider(node_number, relatrive_address):
        if node_number == 0:
            return left_weight_provider(p, n, index, alpha)
        elif node_number == p:
            return right_weight_provider(p, n, index, alpha)
        else:
            j = node_number
            return interior_weights_provider(p, j, index)

    def weights_provider(*args):
        return _multiplier * pure_weight_provider(*args)

    return Stencil.uniform(Point(start), Point(end), p, weights_provider)


def create_left_caputo_stencil(alpha, lf, p):
    return create_side_caputo_stencil(
        alpha, p,
        -lf,
        0.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, n, index, alpha: 1.,
        lambda p, j, index: (p - j + 1.) ** index - 2. * (p - j) ** index + (p - j - 1.) ** index,
        lambda n: 1.,
    )


def create_right_caputo_stencil(alpha, lf, p):
    return create_side_caputo_stencil(
        alpha, p,
        0.,
        lf,
        lambda p, n, index, alpha: 1.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, j, index: (j + 1.) ** index - 2. * j ** index + (j - 1.) ** index,
        lambda n: (-1.) ** n,
    )


def create_riesz_caputo_stencil(settings):
    alpha, lf, resolution = settings
    n = math.floor(alpha) + 1.

    left = create_left_caputo_stencil(alpha, lf, resolution)
    right = create_right_caputo_stencil(alpha, lf, resolution)

    return Number(1. / 2. * math.gamma(2. - alpha) / math.gamma(2.)) * \
           (left + Number((-1.) ** n) * right)


