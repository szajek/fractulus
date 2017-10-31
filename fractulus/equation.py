import collections
import math

from fdm.equation import Stencil, Number, DynamicStencil

__all__ = ['CaputoSettings', 'create_left_caputo_stencil', 'create_right_caputo_stencil', 'create_riesz_caputo_stencil']


CaputoSettings = collections.namedtuple('CaputoSettings', ('alpha', 'lf', 'resolution'))


def create_side_caputo_stencil(alpha, p, left_range, right_range, left_weight_provider, right_weight_provider,
                               interior_weights_provider, multiplier):
    n = math.floor(alpha) + 1.
    index = (n - alpha + 1.)
    _multiplier = multiplier(n) * (1. / math.gamma(n - alpha + 2.))

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

    return Stencil.uniform(left_range, right_range, p, weights_provider, order=-(n - alpha))


def create_left_caputo_stencil(alpha, p):
    return create_side_caputo_stencil(
        alpha, p,
        p,
        0.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, n, index, alpha: 1.,
        lambda p, j, index: (p - j + 1.) ** index - 2. * (p - j) ** index + (p - j - 1.) ** index,
        lambda n: 1.,
    )


def create_right_caputo_stencil(alpha, p):
    return create_side_caputo_stencil(
        alpha, p,
        0.,
        p,
        lambda p, n, index, alpha: 1.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, j, index: (j + 1.) ** index - 2. * j ** index + (j - 1.) ** index,
        lambda n: (-1.) ** n,
    )


def create_riesz_caputo_stencil(settings, increase_order_by=0., dynamic_resolution=None):
    alpha = settings.alpha
    n = math.floor(alpha) + 1.

    def left_caputo_stencil_builder(node_address):
        dynamic_p = dynamic_resolution(node_address) if dynamic_resolution is not None else settings.resolution
        stencil = create_left_caputo_stencil(alpha, dynamic_p)
        mutated = stencil.mutate(order=stencil.order + increase_order_by)
        return mutated

    def right_caputo_stencil_builder(node_address):
        dynamic_p = dynamic_resolution(node_address) if dynamic_resolution else settings.resolution
        stencil = create_right_caputo_stencil(alpha, dynamic_p)
        mutated = stencil.mutate(order=stencil.order + increase_order_by)
        return mutated

    left_stencil = DynamicStencil(left_caputo_stencil_builder)
    right_stencil = DynamicStencil(right_caputo_stencil_builder)

    return Number(1. / 2. * math.gamma(2. - alpha) / math.gamma(2.)) * \
           (left_stencil + Number((-1.) ** n) * right_stencil)


