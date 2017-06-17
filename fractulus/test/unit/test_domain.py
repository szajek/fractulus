import unittest

from fractulus.domain import Grid1DBuilder, Grid, Node, Connection
from mock import MagicMock


class GridTest(unittest.TestCase):
    def test_GetConnections_NotExtremeNode_ReturnBackwardAndForwardConnection(self):

        connections, grid = self._create_3node_grid()

        result = grid.get_connections(1)

        expected = connections

        self.assertEquals(expected, result)

    def test_GetConnections_FirstNode_ReturnOnlyForwardConnection(self):

        connections, grid = self._create_3node_grid()

        result = grid.get_connections(0)

        expected = connections[:1]

        self.assertEquals(expected, result)

    def test_GetConnections_LastNode_ReturnOnlyBackwardConnection(self):

        connections, grid = self._create_3node_grid()

        result = grid.get_connections(2)

        expected = connections[1:]

        self.assertEquals(expected, result)

    def test_GetByAddress_Exists_ReturnNode(self):
        grid = Grid(
            nodes=[
                MagicMock(tag='0'),
                MagicMock(tag='1'),
                MagicMock(tag='2'),
            ],
            connections=[]
        )

        for i in range(3):
            self.assertEqual(str(i), grid.get_by_address(i).tag)

    def _create_3node_grid(self):
        nodes = [
            Node(0.),
            Node(1.),
            Node(3.),
        ]
        connections = [
            Connection(*nodes[0: 2]),
            Connection(*nodes[1: 3]),
        ]
        grid = Grid(nodes, connections)
        return connections, grid


class Domain1DBuilderTest(unittest.TestCase):
    def test_AddUniformlyDistributedNodes_NumberIsZeroOrOne_Raise(self):

        b = Grid1DBuilder(4)

        with self.assertRaises(AttributeError):
            b.add_uniformly_distributed_nodes(1)

    def test_AddUniformlyDistributedNodes_MoreThanOne_CreateGivenNumberOfEquallySpacedNodes(self):
        node_number = 3

        nodes, connections = self._create_uniformly_distributed_nodes(node_number)

        self.assertEqual(node_number, len(nodes))

        self.assertEqual(1., len(set(self._calculate_spaces_between_nodes(nodes))))

    def test_AddUniformlyDistributedNodes_MoreThanOne_CreateConnectionsForSuccessiveNodes(self):
        node_number = 3

        nodes, connections = self._create_uniformly_distributed_nodes(node_number)

        for i in range(node_number - 1):
            self.assertEqual(nodes[i: i+2], list(connections[i]))

    def _calculate_spaces_between_nodes(self, nodes):
        def calc_distance_to_predecessor(i):
            return nodes[i].x - nodes[i - 1].x

        return [calc_distance_to_predecessor(i) for i in range(1, len(nodes))]

    def _create_uniformly_distributed_nodes(self, node_number):
        domain = (
            Grid1DBuilder(6)
                .add_uniformly_distributed_nodes(node_number)
                .create()
        )
        return domain.nodes, domain.connections

