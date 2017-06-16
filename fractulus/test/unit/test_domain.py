import unittest

from fractulus.domain import Grid1DBuilder, Grid, Node, Connection
from mock import MagicMock


class NodeTest(unittest.TestCase):
    pass


class SectionTest(unittest.TestCase):
    pass


class GridTest(unittest.TestCase):
    def test_GetConnections_MiddleNode_ReturnBackwardAndForwardConnection(self):

        connections, grid = self._create_3node_grid()

        result = grid.get_connections(1)

        self.assertEquals(connections, result)

    def test_GetConnections_FirstNode_ReturnOnlyForwardConnection(self):

        connections, grid = self._create_3node_grid()

        result = grid.get_connections(0)

        self.assertEquals([connections[0]], result)

    def test_GetConnections_LastNode_ReturnOnlyBackwardConnection(self):

        connections, grid = self._create_3node_grid()

        result = grid.get_connections(2)

        self.assertEquals([connections[1]], result)

    def test_GetByAddress_Exists_ReturnNode(self):
        grid = Grid(
            nodes=[
                MagicMock(tag='0'),
                MagicMock(tag='1'),
                MagicMock(tag='2'),
            ],
            connections=[]
        )

        self.assertEqual('0', grid.get_by_address(0).tag)

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
    def test_NodesByNumber_ZeroOrOne_Raise(self):
        b = Grid1DBuilder(4)
        with self.assertRaises(AttributeError):
            b.nodes_by_number(1)

    def test_NodesByNumber_MoreThan1_CreateGivenNodesNumberUniformlyDistributed(self):
        b = Grid1DBuilder(6)
        b.nodes_by_number(3)
        domain = b.create()

        nodes = domain.nodes
        self.assertEqual(3, len(nodes))
        self.assertEqual(0., nodes[0].x)
        self.assertEqual(3., nodes[1].x)
        self.assertEqual(6., nodes[2].x)

    def test_NodesByNumber_MoreThan1_CreateConnectionsForSuccessiveNodes(self):
        b = Grid1DBuilder(6)
        b.nodes_by_number(3)
        domain = b.create()

        nodes = domain.nodes
        connections = domain.connections
        self.assertEqual(nodes[0], connections[0].start)
        self.assertEqual(nodes[1], connections[0].end)
        self.assertEqual(nodes[1], connections[1].start)
        self.assertEqual(nodes[2], connections[1].end)

