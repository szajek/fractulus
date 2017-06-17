

def subtract_or_return_zero(value_1, value_2):
    return 0. if None in [value_1, value_2] else (value_1 - value_2)


class Point:
    def __init__(self, x, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z


class Vector:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def length(self):
        dx = self.end.x - self.start.x
        dy = subtract_or_return_zero(self.end.y, self.start.y)
        dz = subtract_or_return_zero(self.end.z, self.start.z)
        return (dx**2 + dy**2 + dz**2)**.5

    def __iter__(self):
        return [self.start, self.end].__iter__()


class BoundaryBox:
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max

    @property
    def dimensions(self):
        used = self.directions
        return tuple(map(lambda i: self._calculate_dimension(i) if i in used else None, range(len(self.min))))

    @property
    def directions(self):
        return [i for i, (_min, _max) in enumerate(zip(self.min, self.max)) if None not in [_min, _max]]

    def _calculate_dimension(self, direction):
        return self.max[direction] - self.min[direction]

    @classmethod
    def from_points(cls, points):
        _min, _max = calculate_extreme_coordinates(points)
        return cls(_min, _max)


def calculate_extreme_coordinates(points):

    def extreme_or_nones(coordinates):
        return [None, None] if None in coordinates else (min(coordinates), max(coordinates))

    def extract_coordinates(extractor):
        return [extractor(node) for node in points]

    def create_extractor(coord_name):
        return lambda node: getattr(node, coord_name)

    return tuple(
        zip(*[
            extreme_or_nones(
                extract_coordinates(
                    create_extractor(coord_name)
                )
            ) for coord_name in ['x', 'y', 'z']
            ])
    )
