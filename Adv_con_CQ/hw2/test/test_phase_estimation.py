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


class TestPhaseEstimation(unittest.TestCase):

    def test(self):
        """
        Test application of phase estimation circuit.
        """
        t = 3
        n = 4
        for j in range(2**t):
            # to-be estimated phase
            ϕ = j / 2**t
            # random base change matrix
            Q = unitary_group.rvs(2**n)
            # phases stored in `U`
            λ = np.array(np.insert(np.random.uniform(0, 1, 2**n - 1), 0, ϕ))
            U = Q @ np.diag(np.exp(2*np.pi*1j * λ)) @ Q.conj().T
            # eigenstate of U (input to phase estimation algorithm)
            v = Q[:, 0]
            self.assertTrue(np.allclose(U @ v, np.exp(2*np.pi*1j * ϕ) * v))
            ψphase = cs.phase_estimation_circuit(v, U, t, n)
            # reference unit vector
            ψphase_ref = np.zeros(2**t)
            ψphase_ref[j] = 1
            self.assertTrue(np.allclose(ψphase, ψphase_ref))


if __name__ == '__main__':
    unittest.main()
