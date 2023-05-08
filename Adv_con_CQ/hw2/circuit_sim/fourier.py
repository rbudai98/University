from .gates import Gates, apply_gate, apply_controlled_gate
import numpy

def quantum_fourier_transform_three_qubits(ψ, nqubits):
    """
    Three-qubit Fourier transform applied to the leading qubits.
    """
    # first Hadamard gate
    ψ = apply_gate(ψ, Gates.H, 0, nqubits)
    # first controlled-S gate
    ψ = apply_controlled_gate(ψ, 1, Gates.S, 0, nqubits)
    # controlled-T gate
    ψ = apply_controlled_gate(ψ, 2, Gates.T, 0, nqubits)
    # second Hadamard gate
    ψ = apply_gate(ψ, Gates.H, 1, nqubits)
    # second controlled-S gate
    ψ = apply_controlled_gate(ψ, 2, Gates.S, 1, nqubits)
    # third Hadamard gate
    ψ = apply_gate(ψ, Gates.H, 2, nqubits)
    # final swap gate
    ψ = apply_gate(ψ, Gates.swap, (0, 2), nqubits)
    return ψ

def quantum_fourier_transform(ψ, n, nqubits):
    """
    Fourier transform circuit applied to the leading `n` qubits.
    """
    for j in range(n):
        ψ = apply_gate(ψ, Gates.H, j, nqubits)
        for k in range(j+1, n):
            ψ = apply_controlled_gate(ψ, k, Gates.R(2**((-1)*(k - j + 1))), j, nqubits)
    # TODO: apply the swap gates
    for i in range(int(n/2)):
        ψ = apply_gate(ψ, Gates.swap,(i, n-i-1),nqubits)
    return ψ


def quantum_inverse_fourier_transform(ψ, n, nqubits):
    """
    Inverse Fourier transform circuit applied to the leading `n` qubits.
    """
    # Inverse circuit results from reversing the ordering of all gates and
    # taking the adjoint of each gate.
    for i in range(int(n/2)):
        ψ = apply_gate(ψ, Gates.swap,(i, n-i-1),nqubits)
    for j in range(n-1, -1, -1):
        for k in range(n-1, j, -1):
            ψ = apply_controlled_gate(ψ, k, Gates.R(-2**((-1)*(k - j + 1))), j, nqubits)       
        ψ = apply_gate(ψ, Gates.H, j, nqubits)  

    return ψ
