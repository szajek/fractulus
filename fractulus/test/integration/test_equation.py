import unittest

import collections

from fractulus.equation import CaputoSettings, create_left_caputo_stencil, create_riesz_caputo_stencil
from fdm import Operator, Number, Stencil


class LeftCaputoStencilTest(unittest.TestCase):
    def test_Expand_NodeZeroAndAlphaAlmostOne_ReturnNodeZeroWeightAlmostOne(self):

        settings = CaputoSettings(0.9999, 1., 1.)
        stencil = create_left_caputo_stencil(settings)

        self.assertAlmostEqual(
            1.,
            stencil.expand(0)._weights[0],
            places=3,
        )


