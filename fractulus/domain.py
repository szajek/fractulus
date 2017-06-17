from functools import reduce
from operator import is_not
from functools import partial

from .geometry import Point, Vector, calculate_boundary_box, calculate_dimensions


__all__ = ['Node', 'Connection', 'Grid', 'Grid1DBuilder', ]


def reduce_or_return_none(function, sequence, initial=None):
    try:
        return reduce(function, sequence, initial)
    except StopIteration:
        return None


class Node(Point):
    def __repr__(self):
        return 'x={x}, y={y}, z={z}'.format(x=self.x, y=self.y, z=self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z


class Connection(Vector):
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        return "FROM: %s; TO: %s" % (self.start.__repr__(), self.end.__repr__())


class Grid:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections

    def get_connections(self, node_address):
        backward = None if node_address <= 0 else self.connections[node_address - 1]
        forward = None if node_address >= len(self.nodes) - 1 else self.connections[node_address]
        return list(filter(partial(is_not, None), [backward, forward]))

    def get_by_address(self, address):
        return self.nodes[int(address)]

    def calculate_boundary_box(self):
        return calculate_boundary_box(self.nodes)

    def get_dimensions(self):
        bbox = self.calculate_boundary_box()
        return calculate_dimensions(bbox)


class Grid1DBuilder:
    def __init__(self, length, start=0.):
        self._length = length
        self._start = start

        self._nodes = []
        self._connections = []

    def add_uniformly_distributed_nodes(self, number):
        if number < 2:
            raise AttributeError("Number of point must be at least 2")
        section_length = self._length / (number - 1)

        prev_node = self.add_node_by_coordinate(self._start)
        for node_num in range(number - 1):
            next_node = self.add_node_by_coordinate(self._start + (node_num + 1) * section_length)
            self.add_connection_by_nodes(prev_node, next_node)
            prev_node = next_node

        return self

    def add_node_by_coordinate(self, coord):
        node = Node(coord)
        self._nodes.append(node)
        return node

    def add_connection_by_nodes(self, start, end):
        connection = Connection(start, end)
        self._connections.append(
            connection
        )
        return connection

    def create(self):
        return Grid(
            self._nodes,
            self._connections,
        )
