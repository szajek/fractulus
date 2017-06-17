import unittest

from fractulus.geometry import Point, Vector, calculate_extreme_coordinates, BoundaryBox


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


class CalculateExtremeCoordinatesTest(unittest.TestCase):
    def test_Call_OnlyXCoordinate_ReturnListsOfExtremesOrNoneInAllDirections(self):
        nodes = [Point(1.), Point(2.), Point(3.)]

        result = calculate_extreme_coordinates(nodes)

        expected = (
           (nodes[0].x, None, None),
           (nodes[-1].x, None, None)
        )

        self.assertEqual(expected, result)


class BoundaryBoxTest(unittest.TestCase):
    def test_Dimensions_OnlyXCoordinate_ReturnXRangeAndNoneForYAndZ(self):
        points = [Point(-1.), Point(5.)]
        bbox = BoundaryBox.from_points(points)

        result = bbox.dimensions

        expected = (6., None, None)

        self.assertEqual(expected, result)

    def test_Directions_OnlyX_ReturnListWithIndexZero(self):
        points = [Point(-1.)]
        bbox = BoundaryBox.from_points(points)

        result = bbox.directions

        expected = [0]

        self.assertEqual(expected, result)

    def test_Directions_XYZ_ReturnListWithThreeIndices(self):
        points = [Point(-1., 3., 3.)]
        bbox = BoundaryBox.from_points(points)

        result = bbox.directions

        expected = [0, 1, 2]

        self.assertEqual(expected, result)