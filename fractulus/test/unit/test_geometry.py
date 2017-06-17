import unittest

from fractulus.geometry import Point, Vector, calculate_boundary_box, calculate_dimensions


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

        expected = 2.55 - 2.

        self.assertAlmostEqual(expected, result)

    def test_Iterate_Always_IterateOverPoints(self):

        nodes = [Point(1.), Point(2.)]
        vector = Vector(*nodes)

        for i, point in enumerate(vector):
            self.assertEqual(nodes[i], point)


class CalculateBoundaryBoxTest(unittest.TestCase):
    def test_Call_OnlyXCoordinate_ReturnListsOfExtremesOrNoneInAllDirections(self):
        nodes = [Point(1.), Point(2.), Point(3.)]

        result = calculate_boundary_box(nodes)

        expected = (
           (nodes[0].x, None, None),
           (nodes[-1].x, None, None)
        )

        self.assertEqual(expected, result)


class CalculateDimensionsTest(unittest.TestCase):
    def test_Call_OnlyXCoordinate_ReturnXRangeAndNoneForYAndZ(self):
        nodes = [
            [-2., None, None],
            [2., None, None]
        ]

        result = calculate_dimensions(nodes)

        expected = (4., None, None)

        self.assertEqual(expected, result)