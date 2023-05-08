import unittest
import numpy as np
from scipy.stats import unitary_group
import circuit_sim as cs


def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


def reorder_gate(G, perm):
    """
    Adapt gate `G` to an ordering of the qubits as specified in `perm`.

    Example, given G = np.kron(np.kron(A, B), C):
        reorder_gate(G, [1, 2, 0]) == np.kron(np.kron(B, C), A)
    """
    perm = list(perm)
    # number of qubits
    n = len(perm)
    # reorder both input and output dimensions
    perm2 = perm + [n + i for i in perm]
    return np.reshape(np.transpose(np.reshape(G, 2*n*[2]), perm2), (2**n, 2**n))


class TestGates(unittest.TestCase):

    def test_apply_1gate(self):
        """
        Test application of a single-qubit gate.
        """
        nqubits = 4
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        # random unitary matrix
        U = unitary_group.rvs(2)
        # apply gate to statevector
        Uψ = cs.apply_gate(ψ, U, 1, nqubits)
        # reference calculation
        Uψref = np.kron(np.kron(np.identity(2), U), np.identity(4)) @ ψ
        # compare
        self.assertTrue(np.allclose(Uψ, Uψref), msg="result of applying a single-qubit gate does not match reference")
        # another random unitary matrix
        V = crandn((2, 2))
        # apply gate to statevector
        Vψ = cs.apply_gate(ψ, V, 3, nqubits)
        # reference calculation
        Vψref = np.kron(np.identity(8), V) @ ψ
        # compare
        self.assertTrue(np.allclose(Vψ, Vψref), msg="result of applying a single-qubit gate does not match reference")

    def test_apply_2gate(self):
        """
        Test application of a two-qubit gate.
        """
        nqubits = 5
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        # random unitary matrix
        U = unitary_group.rvs(4)
        # apply gate to statevector
        Uψ = cs.apply_gate(ψ, U, (1, 4), nqubits)
        # reference calculation
        Uψref = reorder_gate(np.kron(U, np.identity(8)), (2, 0, 3, 4, 1)) @ ψ
        # compare
        self.assertTrue(np.allclose(Uψ, Uψref), msg="result of applying a two-qubit gate does not match reference")

    def test_apply_3gate(self):
        """
        Test application of a three-qubit gate.
        """
        nqubits = 5
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        # random unitary matrix
        U = unitary_group.rvs(8)
        # apply gate to statevector
        Uψ = cs.apply_gate(ψ, U, (0, 2, 3), nqubits)
        # reference calculation
        Uψref = reorder_gate(np.kron(U, np.identity(4)), (0, 3, 1, 2, 4)) @ ψ
        # compare
        self.assertTrue(np.allclose(Uψ, Uψref), msg="result of applying a three-qubit gate does not match reference")

    def test_apply_4gate(self):
        """
        Test application of a four-qubit gate.
        """
        nqubits = 6
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        # random unitary matrix
        U = unitary_group.rvs(16)
        # apply gate to statevector
        Uψ = cs.apply_gate(ψ, U, (0, 2, 3, 5), nqubits)
        # reference calculation
        Uψref = reorder_gate(np.kron(U, np.identity(4)), (0, 4, 1, 2, 5, 3)) @ ψ
        # compare
        self.assertTrue(np.allclose(Uψ, Uψref), msg="result of applying a four-qubit gate does not match reference")

    def test_apply_controlled_gate(self):
        """
        Test application of a controlled single-qubit gate.
        """
        nqubits = 5
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        # random unitary matrix
        U = unitary_group.rvs(4)
        # apply gate to statevector, controlled by qubit 1
        cUψ = cs.apply_controlled_gate(ψ, 1, U, (0, 3), nqubits)
        # reference calculation
        # manually construct controlled-U gate
        cU = np.zeros((8, 8), dtype=complex)
        cU[0:4, 0:4] = np.identity(4)
        cU[4:8, 4:8] = U
        cUψref = reorder_gate(np.kron(cU, np.identity(4)), (1, 0, 3, 2, 4)) @ ψ
        # compare
        self.assertTrue(np.allclose(cUψ, cUψref), msg="result of applying a controlled gate does not match reference")


if __name__ == '__main__':
    unittest.main()
