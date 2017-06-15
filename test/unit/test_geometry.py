import unittest

from geometry import Point, Vector


class PointTest(unittest.TestCase):

    def test_Create_OnlyX_ReturnPointWithNoneYandZ(self):
        p = Point(2.)
        self.assertEqual(None, p.y)
        self.assertEqual(None, p.z)


class VectorTest(unittest.TestCase):

    def test_Length_OnlyX_ReturnDistanceBetweenNodes(self):
        p1 = Point(2.)
        p2 = Point(2.55)

        result = Vector(p1, p2).length

        self.assertAlmostEqual(0.55, result)