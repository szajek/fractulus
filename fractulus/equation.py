import collections
import math

from fdm.equation import Stencil, Number, DynamicStencil

__all__ = ['CaputoSettings']


CaputoSettings = collections.namedtuple('CaputoSettings', ('alpha', 'lf', 'resolution'))


def create_side_caputo_stencil(alpha, p, left_range, right_range, left_weight_provider, right_weight_provider,
                               interior_weights_provider, multiplier):
    # alpha = settings.alpha
    n = math.floor(alpha) + 1.
    index = (n - alpha + 1.)
    # p = settings.resolution
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


def create_left_caputo_stencil(alpha, p, lf):
    return create_side_caputo_stencil(
        alpha, p,
        lf,
        0.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, n, index, alpha: 1.,
        lambda p, j, index: (p - j + 1.) ** index - 2. * (p - j) ** index + (p - j - 1.) ** index,
        lambda n: 1.,
    )


def create_right_caputo_stencil(alpha, p, lf):
    return create_side_caputo_stencil(
        alpha, p,
        0.,
        lf,
        lambda p, n, index, alpha: 1.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, j, index: (j + 1.) ** index - 2. * j ** index + (j - 1.) ** index,
        lambda n: (-1.) ** n,
    )


def create_riesz_caputo_stencil(settings, increase_order_by=0., dynamic_resolution=lambda address: 1.):
    alpha = settings.alpha
    # p = settings.resolution
    lf = settings.lf
    n = math.floor(alpha) + 1.

    def left_caputo_stencil_builder(node_address):
        dynamic_p = dynamic_resolution(node_address)
        stencil = create_left_caputo_stencil(alpha, dynamic_p, lf)
        return stencil.mutate(order=stencil.order + increase_order_by)

    def right_caputo_stencil_builder(node_address):
        dynamic_p = dynamic_resolution(node_address)
        stencil = create_right_caputo_stencil(alpha, dynamic_p, lf)
        return stencil.mutate(order=stencil.order + increase_order_by)

    left_stencil = DynamicStencil(left_caputo_stencil_builder)
    right_stencil = DynamicStencil(right_caputo_stencil_builder)

    # left_stencil = create_left_caputo_stencil(alpha, p, lf)
    # left_stencil = LocalizedStencil(left_stencil.mutate(order=left_stencil.order + increase_order_by), scheme_corrector)
    # right_stencil = create_right_caputo_stencil(alpha, p, lf)
    # right_stencil = LocalizedStencil(right_stencil.mutate(order=right_stencil.order + increase_order_by), scheme_corrector)

    return Number(1. / 2. * math.gamma(2. - alpha) / math.gamma(2.)) * \
           (left_stencil + Number((-1.) ** n) * right_stencil)


