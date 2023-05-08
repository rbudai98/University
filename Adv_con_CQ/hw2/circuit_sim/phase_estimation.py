import numpy as np
from .gates import Gates, apply_gate, apply_controlled_gate
from .fourier import quantum_inverse_fourier_transform


def phase_estimation_circuit(v, U, t: int, n: int):
    """
    Apply the phase estimation circuit.

    Args:
        v: initial state of second register (should be eigenstate of `U` matrix)
        U: unitary matrix appearing in phase estimation algorithm
        t: number of qubits in the first register
        n: number of qubits in the second register

    Returns:
        np.ndarray: output state in first register,
                    after projecting second register onto `v`
    """
    assert np.size(v) == 2**n, "input `v` must be an n-qubit state"
    assert U.shape == (2**n, 2**n), "input `U` must be a unitary matrix acting on `n` qubits"

    # TODO: prepare initial state of t-qubit register (equal superposition state)
    ψt = np.zeros(2**t)
    ψt = Gates.H
    for i in range(t - 1):
        ψt = np.kron(ψt, Gates.H)
    ψt = ψt[:,0]

    # overall quantum state
    ψ = np.kron(ψt, v)

    # TODO: apply controlled-U gates, and square U in each iteration to realize U^(2^j)
    for i in range(t-1, -1, -1):
        ψ = apply_controlled_gate(ψ, i, U, range(t,n+t), n + t)
        U = np.matmul(U, U)

    # inverse Fourier transform of t register
    ψ = quantum_inverse_fourier_transform(ψ, t, t + n)
    # project second register of ψ onto initial `v` state (to retain first register only)
    P = np.kron(np.identity(2**t), np.reshape(v.conj(), (1, 2**n)))
    return P @ ψ
