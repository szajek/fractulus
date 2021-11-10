import collections
import math

from fdm.equation import Stencil, Number
from fdm.geometry import Point

__all__ = 'Settings', 'create_left_stencil', 'create_right_stencil', 'create_riesz_caputo_stencil'


Settings = collections.namedtuple('CaputoSettings', ('alpha', 'lf', 'resolution'))


def create_riesz_caputo_stencil(approximation, settings):
    return _create_riesz_caputo_stencil(
        settings,
        create_left_stencil(approximation, settings)
    )


def _create_riesz_caputo_stencil(settings, left):
    alpha, lf, resolution = settings
    n = math.floor(alpha) + 1.
    element = Number(1. / 2. * math.gamma(2. - alpha) / math.gamma(2.)) * \
              (left + Number((-1.) ** n) * _create_stencil_by_symmetry(left))
    return element.to_stencil(Point(0.))


def create_left_stencil(approximation, settings):
    return _factory[approximation](settings)


def create_right_stencil(approximation, settings):
    return _create_stencil_by_symmetry(
        _factory[approximation](settings)
    )


def _create_stencil_by_symmetry(left_stencil):
    return left_stencil.symmetry(Point(0.)).multiply(-1.)


def _create_left_caputo_stencil(settings):
    alpha, lf, p = settings
    n = math.floor(alpha) + 1.
    index = (n - alpha + 1.)

    def weights(number):
        if number == 0:
            return (p - 1.) ** index - (p - n + alpha - 1.) * p ** (n - alpha)
        elif number == p:
            return 1.
        else:
            j = number
            return (p - j + 1.) ** index - 2. * (p - j) ** index + (p - j - 1.) ** index

    h = (0. - (-lf)) / p
    _multiplier = ((h ** (n - alpha)) / math.gamma(n - alpha + 2.))

    return _create_parametrized_stencil(-lf, 0., p, _multiplier, weights)


def _create_left_rectangle_rule_stencil(settings):
    alpha, lf, resolution = settings
    dx = lf / float(resolution)

    def weight(i):
        k = -resolution + i
        return (-k)**(1. - alpha) - (-k - 1) ** (1. - alpha)

    return _create_parametrized_stencil(
        -lf + dx, 0., resolution - 1,
        (lf / float(resolution)) ** (1. - alpha) / math.gamma(2. - alpha),
        weight)


def _create_parametrized_stencil(start, end, points_number, multiplier, weight):

    def _weight(node_number, relative_address):
        return multiplier * weight(node_number)

    return Stencil.uniform(Point(start), Point(end), points_number, _weight)


def _create_left_trapezoidal_rule_stencil(settings):
    alpha, lf, p = settings

    def weight(number):

        if number == 0:
            return (p - 1.)**(2 - alpha) + (2. - alpha - p)*p**(1 - alpha)
        elif number == p:
            return 1.
        else:
            k = -p + number
            return (-k + 1.)**(2 - alpha) - 2*(-k)**(2 - alpha) + (-k - 1.)**(2 - alpha)

    return _create_parametrized_stencil(
        -lf, 0., p,
        (lf / float(p)) ** (1. - alpha) / math.gamma(3. - alpha),
        weight)


def _create_left_simpson_rule_stencil(settings):
    alpha, lf, m = settings
    dt = lf / float(m)
    i = 0

    m_is_even = math.fmod(m, 2.) == 0.

    def _calc_k(number):
        return i - m + number

    def _weight_k_eq_i_sub_m():
        a = (m ** (3 - alpha) - (m - 2) ** (3 - alpha)) / (math.gamma(4 - alpha))
        b = - (3 * m ** (2 - alpha) + (m - 2) ** (2 - alpha)) / (2 * math.gamma(3 - alpha))
        c = (m ** (1 - alpha)) / (math.gamma(2 - alpha))
        return a + b + c

    def _weight_odd_k_add_i_sub_m(i, k):
        a = -2 * ((i - k + 1) ** (3 - alpha) - (i - k - 1) ** (3 - alpha)) / (math.gamma(4 - alpha))
        b = 2 * ((i - k + 1) ** (2 - alpha) + (i - k - 1) ** (2 - alpha)) / (math.gamma(3 - alpha))
        return a + b

    def _weight_even_k_add_i_sub_m(i, k):
        a = ((i - k + 2) ** (3 - alpha) - (i - k - 2) ** (3 - alpha)) / (math.gamma(4 - alpha))
        b = ((i - k + 2) ** (2 - alpha) + 6 * (i - k) ** (2 - alpha) + (i - k - 2) ** (2 - alpha)) / (
        2 * math.gamma(3 - alpha))
        return a - b

    def _weight_even_m_and_k_eq_i():
        a = (2 ** (3 - alpha)) / (math.gamma(4 - alpha))
        b = -(2 ** (2 - alpha)) / (2 * math.gamma(3 - alpha))
        return a + b

    def _weight_odd_m_and_k_eq_i_sub_1():
        a = (3**(3 - alpha) - 1)/(math.gamma(4. - alpha))
        b = -(3**(2 - alpha) + 3)/(2*math.gamma(3. - alpha))
        c = -1./(math.gamma(2. - alpha))
        return a + b + c

    def w_i_k(number):
        k = _calc_k(number)
        if k == i - m:
            return _weight_k_eq_i_sub_m()
        elif m_is_even and k == i:
            return _weight_even_m_and_k_eq_i()
        elif not m_is_even and k == i - 1:
            return _weight_odd_m_and_k_eq_i_sub_1()
        elif math.fmod(i - m + k, 2) == 0. and k < i - 1:
            return _weight_even_k_add_i_sub_m(i, k)
        elif math.fmod(i - m + k, 1) == 0.:
            return _weight_odd_k_add_i_sub_m(i, k)
        else:
            return 0.

    def u_k(k):
        if k == -1:
            return (2.*(1. - alpha)**2 + 3. - 3.*alpha)/(2*math.gamma(4. - alpha))
        elif k == 0:
            return (4. - 2.*alpha)/(math.gamma(4. - alpha))
        elif k == 1:
            return (-1. + alpha)/(2.*math.gamma(4. - alpha))
        else:
            return 0.

    def weights_odd(number):
        k = _calc_k(number)
        w = w_i_k(number) if k <= i - 1 else 0.
        u = u_k(k) if k in [-1, 0, 1] else 0.
        return w + u

    multiplier = dt ** (1. - alpha)
    if m_is_even:
        return _create_parametrized_stencil(-lf, 0., m, multiplier, w_i_k)
    else:
        return _create_parametrized_stencil(-lf, dt, m + 1, multiplier, weights_odd)


_factory = {
    'caputo': _create_left_caputo_stencil,
    'rectangle': _create_left_rectangle_rule_stencil,
    'trapezoidal': _create_left_trapezoidal_rule_stencil,
    'simpson': _create_left_simpson_rule_stencil,
}



