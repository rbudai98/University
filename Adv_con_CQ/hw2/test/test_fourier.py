import unittest
import numpy as np
import circuit_sim as cs


def fourier_matrix(n):
    """
    Construct the unitary Fourier transformation matrix.
    """
    return np.array([[np.exp(2*np.pi*1j*j*k/n)/np.sqrt(n) for j in range(n)] for k in range(n)])


def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


class TestFourier(unittest.TestCase):

    def test_three_qubit_fourier_transform(self):
        """
        Test application of the three-qubit quantum Fourier transform circuit.
        """
        nqubits = 5
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        # apply three-qubit Fourier transform circuit to the leading three qubits
        ψF = cs.quantum_fourier_transform_three_qubits(ψ, nqubits)
        # reference calculation
        ψFref = np.kron(fourier_matrix(8), np.identity(4)) @ ψ
        # compare
        self.assertTrue(np.allclose(ψF, ψFref))

    def test_fourier_transform(self):
        """
        Test application of the quantum Fourier transform circuit.
        """
        # total number of qubits
        nqubits = 7
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        for n in range(1, nqubits):
            # apply Fourier transform circuit to the leading `n` qubits
            ψF = cs.quantum_fourier_transform(ψ, n, nqubits)
            # reference calculation
            ψFref = np.kron(fourier_matrix(2**n), np.identity(2**(nqubits - n))) @ ψ
            # compare
            self.assertTrue(np.allclose(ψF, ψFref))

    def test_inverse_fourier_transform(self):
        """
        Test application of the quantum inverse Fourier transform circuit.
        """
        # total number of qubits
        nqubits = 7
        # random statevector
        ψ = crandn(2**nqubits)
        ψ /= np.linalg.norm(ψ)
        for n in range(1, nqubits):
            # apply Fourier transform circuit to the leading `n` qubits
            ψF = cs.quantum_fourier_transform(ψ, n, nqubits)
            # apply inverse Fourier transform circuit
            ψ2 = cs.quantum_inverse_fourier_transform(ψF, n, nqubits)
            # compare: should recover original state
            self.assertTrue(np.allclose(ψ2, ψ))


if __name__ == '__main__':
    unittest.main()
