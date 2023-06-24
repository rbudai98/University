# Transpile unitary matrices
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
import numpy as np
# circuit = QuantumCircuit(1)
# circuit.ry(Parameter('theta'), 0)
# rxrz_basis = transpile(circuit, basis_gates=['rx', 'rz', 'rz'])
# rxrz_basis.draw()

op1 = np.array([[1, 0],[0, -1]])
op2 = np.array([[1, 0],[0, -1]])
op3 = np.array([[1, 0],[0, -1]])

circ = QuantumCircuit(2)
circ.unitary(op1, [0, 1], 'label_1')
circ.unitary(op2, [0, 1], 'label_2')
circ.unitary(op3, [0, 1], 'label_3')

circ.decompose(gates_to_decompose = 'label_2').draw('mpl')