

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
