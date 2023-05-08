import numpy as np


class Gates(object):
    """
    Collection of common quantum gates.
    """
    # Pauli matrices
    X = np.array([[ 0., 1. ], [ 1.,  0.]])
    Y = np.array([[ 0.,-1.j], [ 1.j, 0.]])
    Z = np.array([[ 1., 0. ], [ 0., -1.]])
    # Hadamard gate
    H = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    # S gate
    S = np.array([[1., 0.], [0., 1.j]])
    # T gate
    T = np.array([[1., 0.], [0., np.sqrt(1j)]])
    # swap gate
    swap = np.identity(4)[[0, 2, 1, 3]]

    def R(θ):
        """
        Z-rotation gate used in the Fourier transform circuit.
        """
        return np.array([[1., 0.], [0., np.exp(2*np.pi*1j*θ)]])


def apply_gate(ψ, U, idx, nqubits):
    """
    Apply the (single or multi)-qubit gate `U` to the qubits `idx`.
    """
    assert np.size(ψ) == 2**nqubits
    # convert a single integer to a tuple storing this integer
    if isinstance(idx, int): idx = (idx,)
    assert U.shape == (2**len(idx), 2**len(idx))
    if len(idx) == 1:
        # single-qubit gate
        i = idx[0]
        assert 0 <= i < nqubits
        ψ = np.reshape(ψ, (2**i, 2, 2**(nqubits-i-1)))
        ψ = np.einsum(U, (1, 2), ψ, (0, 2, 3), (0, 1, 3))
        ψ = np.reshape(ψ, -1)
        return ψ
    elif len(idx) == 2:
        # two-qubit gate
        i, j = idx  # unpack indices
        assert 0 <= i < j < nqubits
        U = np.reshape(U, (2, 2, 2, 2))
        ψ = np.reshape(ψ, (2**i, 2, 2**(j-i-1), 2, 2**(nqubits-j-1)))
        ψ = np.einsum(U, (1, 5, 2, 4), ψ, (0, 2, 3, 4, 6), (0, 1, 3, 5, 6))
        ψ = np.reshape(ψ, -1)
        return ψ
    elif len(idx) == 3:
        # three-qubit gate
        i, j, k = idx  # unpack indices
        assert 0 <= i < j < k < nqubits
        U = np.reshape(U, (2, 2, 2, 2, 2, 2))
        ψ = np.reshape(ψ, (2**i, 2, 2**(j-i-1), 2, 2**(k-j-1), 2, 2**(nqubits-k-1)))
        ψ = np.einsum(U, (1, 4, 8, 2, 5, 7), ψ, (0, 2, 3, 5, 6, 7, 9), (0, 1, 3, 4, 6, 8, 9))
        ψ = np.reshape(ψ, -1)
        return ψ
    elif len(idx) == 4:
        # four-qubit gate
        i, j, k, l = idx  # unpack indices
        assert 0 <= i < j < k < l < nqubits
        U = np.reshape(U, (2, 2, 2, 2, 2, 2, 2, 2))
        ψ = np.reshape(ψ, (2**i, 2, 2**(j-i-1), 2, 2**(k-j-1), 2, 2**(l-k-1), 2, 2**(nqubits-l-1)))
        ψ = np.einsum(U, (1, 4, 8, 11, 2, 5, 7, 10), ψ, (0, 2, 3, 5, 6, 7, 9, 10, 12), (0, 1, 3, 4, 6, 8, 9, 11, 12))
        ψ = np.reshape(ψ, -1)
        return ψ
    else:
        raise RuntimeError("currently only up to 4-qubit gates supported")


def apply_controlled_gate(ψ, c, U, idx, nqubits):
    """
    Apply the gate `U` to the qubits in `idx`, controlled by the c-th qubit.
    """
    assert 0 <= c < nqubits
    # convert a single integer to a tuple storing this integer
    if isinstance(idx, int): idx = (idx,)
    assert not c in idx, "control and target qubits must be different"
    ψ = ψ.copy()
    ψ = np.reshape(ψ, (2**c, 2, 2**(nqubits-c-1)))
    idx_eff = [i if i < c else i - 1 for i in idx]
    # apply U to statevector entries for which the control qubit is in the 1 state
    ψ[:, 1, :] = np.reshape(apply_gate(ψ[:, 1, :], U, idx_eff, nqubits - 1),
                            (2**c, 2**(nqubits-c-1)))
    ψ = np.reshape(ψ, -1)
    return ψ
