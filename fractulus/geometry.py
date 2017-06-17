

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


def calculate_boundary_box(nodes):

    def extreme_or_nones(coordinates):
        return [None, None] if None in coordinates else (min(coordinates), max(coordinates))

    def extract_coordinates(extractor):
        return [extractor(node) for node in nodes]

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


def calculate_dimensions(bbox):
    def calculate_dimension(_min, _max):
        return None if None in [_min, _max] else _max - _min

    return tuple(map(calculate_dimension, *bbox))
