import collections
import math

from fdm.equation import Operator, Stencil, Number

__all__ = ['create_fractional_deformation_operator', 'CaputoSettings']


CaputoSettings = collections.namedtuple('CaputoSettings', ('alpha', 'lf', 'resolution'))


def create_side_caputo_stencil(settings, left_range, right_range, left_weight_provider, right_weight_provider,
                               interior_weights_provider, multiplier):
    alpha = settings.alpha
    n = math.floor(alpha) + 1.
    index = (n - alpha + 1.)
    p = settings.resolution
    _multiplier = multiplier(n) * (1. / math.gamma(n - alpha + 2.))

    def pure_weight_provider(node_number, relatrive_address):
        if node_number == 0:
            return left_weight_provider(p, n, index, alpha)
        elif node_number == settings.resolution:
            return right_weight_provider(p, n, index, alpha)
        else:
            j = node_number
            return interior_weights_provider(p, j, index)

    def weights_provider(*args):
        return _multiplier * pure_weight_provider(*args)

    return Stencil.uniform(left_range, right_range, settings.resolution, weights_provider, order=-(n - alpha))


def create_left_caputo_stencil(settings):
    return create_side_caputo_stencil(
        settings,
        settings.lf,
        0.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, n, index, alpha: 1.,
        lambda p, j, index: (p - j + 1.) ** index - 2. * (p - j) ** index + (p - j - 1.) ** index,
        lambda n: 1.,
    )


def create_right_caputo_stencil(settings):
    return create_side_caputo_stencil(
        settings,
        0.,
        settings.lf,
        lambda p, n, index, alpha: 1.,
        lambda p, n, index, alpha: (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha),
        lambda p, j, index: (j + 1.) ** index - 2. * j ** index + (j - 1.) ** index,
        lambda n: (-1.) ** n,
    )


def create_riesz_caputo_stencil(settings, increase_order_by=0.):
    alpha = settings.alpha
    n = math.floor(alpha) + 1.

    left_stencil = create_left_caputo_stencil(settings)
    left_stencil = left_stencil.mutate(order=left_stencil.order + increase_order_by)
    right_stencil = create_right_caputo_stencil(settings)
    right_stencil = right_stencil.mutate(order=right_stencil.order + increase_order_by)

    return Number(1. / 2. * math.gamma(2. - alpha) / math.gamma(2.)) * \
           (left_stencil + Number((-1.) ** n) * right_stencil)


def create_fractional_deformation_operator(settings):
    alpha = settings.alpha
    n = math.floor(alpha) + 1.
    p = settings.resolution

    multiplier = p ** (alpha - 1)  # l_ef**(alpha-1) = p**(alpha-1) * 1./h**(1-alpha) <- lef describes by grid span

    return Operator(
        Number(multiplier) * create_riesz_caputo_stencil(settings, increase_order_by=1-alpha),
        Operator(  # todo: replace with Operator.order(n)
            Stencil.central(1.),
        )
    )
