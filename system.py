import abc
import collections
import enum
import itertools

import numpy as np
import sys

from finite_difference import Delta

__all__ = ['LinearEquation', 'solve', 'VirtualValueStrategy']

LinearEquation = collections.namedtuple('LinearEquation', ('coefficients', 'free_value'))


class VirtualValueStrategy(enum.Enum):
    SYMMETRY = 0
    AS_IN_BORDER = 1


def extract_virtual_nodes(equation, domain, strategy):
    nodes_number = len(domain.nodes)
    last_node_idx = (nodes_number - 1)

    def detect_node_location(node_id):
        return -1 if node_id < 0. else 1. if node_id >= nodes_number else 0.

    def find_symmetric_node(node_id, location):
        return abs(node_id) if location == -1 else last_node_idx - (node_id - last_node_idx)

    def find_boundary_node(location):
        return 0 if location == -1 else last_node_idx

    def find_corresponding_node(node_id, location):
        if strategy == VirtualValueStrategy.SYMMETRY:
            return find_symmetric_node(node_id, location)
        elif strategy == VirtualValueStrategy.AS_IN_BORDER:
            return find_boundary_node(location)
        else:
            raise NotImplementedError

    def create_virtual_node_if_needed(node_id):
        location = detect_node_location(node_id)
        if location:
            return VirtualNode(node_id, find_corresponding_node(node_id, location))

    return list(filter(None, [create_virtual_node_if_needed(node_id) for node_id in equation.coefficients.keys()]))


def model_to_equations(model):

    def create_equation(node_address):
        if node_address in model.bcs:
            bc = model.bcs[node_address]
            free_value = model.equation.free_value(node_address) if '--bc-no-for-free' in sys.argv else bc.free_value(node_address)
            return LinearEquation(
                bc.coefficients.expand(node_address).to_coefficients(1.),  # todo: make delta not necessary
                free_value)
        else:
            delta = Delta.from_connections(*model.domain.get_connections(node_address))
            return LinearEquation(
                    model.equation.weights(node_address).to_coefficients(delta),
                    model.equation.free_value(node_address)
            )

    return [create_equation(i) for i, node in enumerate(model.domain.nodes)]


VirtualNode = collections.namedtuple('VirtualNode', ('address', 'corresponding_address', ))


class Writer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_coefficients_array(self, size):
        raise NotImplementedError

    @abc.abstractmethod
    def to_free_value(self):
        raise NotImplementedError

    def _create_row(self, size):
        return np.zeros(size)


class EquationWriter(Writer):
    def __init__(self, equation, renumerator):
        self._equation = equation
        self._renumerator = renumerator

    def to_coefficients_array(self, size):
        row = self._create_row(size)
        for variable_number, coefficient in self._equation.coefficients.items():
            variable_number = self._renumerator.get(variable_number, variable_number)
            row[int(variable_number)] = coefficient
        return row

    def to_free_value(self):
        return self._equation.free_value


class VirtualNodeWriter(Writer):
    def __init__(self, virtual_node, virtual_node_number, real_variable_number):
        self._virtual_node = virtual_node
        self._virtual_node_number = virtual_node_number
        self._real_variable_number = real_variable_number

    def to_coefficients_array(self, size):
        row = self._create_row(size)
        variable_number = self._real_variable_number + self._virtual_node_number
        row[variable_number] = 1.
        row[int(self._virtual_node.corresponding_address)] = -1.
        return row

    def to_free_value(self):
        return 0.


class Output(collections.Mapping):

    @property
    def real(self):
        return self._real_output

    def __getitem__(self, key):
        if self._address_forwarder.get(key, key) >= len(self._full_output):
            pass
        return self._full_output[self._address_forwarder.get(key, key)]

    def __iter__(self):
        return self._real_output.__iter__()

    def __len__(self):
        return len(self._real_output)

    def __init__(self, full_output, variable_number, address_forwarder):
        self._full_output = full_output
        self._address_forwarder = address_forwarder

        self._real_output = full_output[:variable_number]
        self._virtual_output = full_output[variable_number:]

    def __repr__(self):
        return "{name}: real: {real}; virtual: {virtual}".format(
            name=self.__class__.__name__, real=self._real_output, virtual=self._virtual_output)


def _solve(solver, model, strategy=VirtualValueStrategy.SYMMETRY):
    equations = model_to_equations(model)

    real_variables_number = len(equations)

    def create_address_forwarder(virtual_nodes):
        return collections.OrderedDict(
            [(vn.address, real_variables_number + i) for i, vn in enumerate(virtual_nodes)])

    def create_writers(_virtual_nodes):

        equations_writers = map(lambda eq: EquationWriter(eq, address_forwarder), equations)
        virtual_node_writers = map(lambda i, vn: VirtualNodeWriter(vn, i, real_variables_number),
                                   *zip(*enumerate(_virtual_nodes))) if _virtual_nodes else []
        return itertools.chain(equations_writers, virtual_node_writers)

    def create_virtual_nodes():
        return sum([extract_virtual_nodes(equation, model.domain, strategy) for equation in equations], [])

    def prepare_empty_arrays(variables_number):
        return np.zeros((variables_number, variables_number)), np.zeros(variables_number)

    def create_arrays():
        variable_number = real_variables_number + len(virtual_nodes)

        weights_array, free_vector_array = prepare_empty_arrays(variable_number)
        for i, writer in enumerate(create_writers(virtual_nodes)):
            weights_array[i] = writer.to_coefficients_array(variable_number)
            free_vector_array[i] = writer.to_free_value()
        return weights_array, free_vector_array

    virtual_nodes = create_virtual_nodes()
    address_forwarder = create_address_forwarder(virtual_nodes)
    return Output(solver(*create_arrays()), real_variables_number, address_forwarder)


def create_linear_system_of_equations_solver():
    def _solve(A, b):
        return np.linalg.solve(A, b[np.newaxis].T)
    return _solve


def create_eigenproblem_solver():
    def _solve(A, b):
        mass = np.diag(np.ones(b.size))
        matrix = np.dot(A, np.linalg.inv(mass))
        eval, evect = np.linalg.eig(matrix)
        return evect[:, 0]
    return _solve


_solvers = {
    'linear_system_of_equations': create_linear_system_of_equations_solver(),
    'eigenproblem': create_eigenproblem_solver(),
}


def solve(solver, *args, **kwargs):
    return _solve(_solvers[solver], *args, **kwargs)