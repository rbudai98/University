"""Toy code implementing the time evolving block decimation (TEBD)."""

import numpy as np
from scipy.linalg import expm
from a_mps import split_truncate_theta


def calc_U_bonds(model, dt):
    """Given a model, calculate ``U_bonds[i] = expm(-dt*model.H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        H = np.reshape(H, [d * d, d * d])
        U = expm(-dt * H)
        U_bonds.append(np.reshape(U, [d, d, d, d]))
    return U_bonds


def run_TEBD(psi, U_bonds, N_steps, chi_max, eps):
    """Evolve the state `psi` for `N_steps` time steps with (first order) TEBD.

    The state psi is modified in place."""
    Nbonds = psi.L - 1
    assert len(U_bonds) == Nbonds
    for n in range(N_steps):
        for k in [0, 1]:  # even, odd
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)
    # done


def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = i + 1
    # construct theta matrix
    theta = psi.get_theta2(i)  # vL i j vR
    # apply U
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    # split and truncate
    Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
    # put back into MPS
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
    psi.Ss[j] = Sj  # vC
    psi.Bs[j] = Bj  # vC j vR



def example_TEBD_gs_finite(L, J, g):
    print("finite TEBD, (imaginary time evolution)")
    print("L={L:d}, J={J:.1f}, g={g:.2f}".format(L=L, J=J, g=g))
    import a_mps
    import b_model
    model = b_model.TFIModel(L, J=J, g=g)
    psi = a_mps.init_spinup_MPS(L)
    for dt in [0.1, 0.01, 0.001, 1.e-4, 1.e-5]:
        U_bonds = calc_U_bonds(model, dt)
        run_TEBD(psi, U_bonds, N_steps=500, chi_max=100, eps=1.e-10)
        E = model.energy(psi)
        print("dt = {dt:.5f}: E = {E:.13f}".format(dt=dt, E=E))
    print("final bond dimensions: ", psi.get_chi())
    return E, psi, model


def calc_correlation_func(op1, op2, psi, site):
    L = psi.L
    contraction = psi.get_theta1(site)

    result = []
    result.append(
    psi.site_expectation_value(
        np.tensordot(op1, op2, (1,0))
        )[site]
    )

    for j in range(site + 1, L, 1):
        # init contraction again
        contraction = psi.get_theta1(site)
        op_theta = np.tensordot(op1, contraction, axes=[0, 1]) # i [i*], vL [i] vR
        contraction = np.tensordot(contraction.conj(), op_theta, [[2, 0], [1, 2]]) # vR*, vR

        # contract until we arrive at position of op2 (index j)
        for k in range(site + 1, j, 1):
            op_theta = np.tensordot(psi.Bs[k], contraction, [1, 0])
            contraction = np.tensordot(psi.Bs[k].conj(), op_theta, [[2, 0], [1, 0]])

        # contract with op2
        op_theta = np.tensordot(psi.Bs[j], contraction, [1, 0])
        op_theta = np.tensordot(op2, op_theta, [1, 2]) # i [i*], vL [i] vR
        contraction = np.tensordot(psi.Bs[j].conj(), op_theta, [[2, 0], [1, 2]])

        # contract until end
        for k in range(j + 1, L, 1):
            op_theta = np.tensordot(psi.Bs[k], contraction, [1, 0])
            contraction = np.tensordot(psi.Bs[k].conj(), op_theta, [[2, 0], [1, 0]])

        result.append(
        np.trace(contraction)
        )

    return np.array(result)
      


if __name__ == "__main__":
    example_TEBD_gs_finite(L=14, J=1., g=1.5)

