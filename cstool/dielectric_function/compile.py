from cslib import units
from cstool.compile import compute_tcs_icdf, compute_2d_tcs_icdf
import numpy as np


def compile_ashley_imfp_icdf(dimfp,
    K, P_omega,
    F):
    """Compute inverse mean free path, and ω' ICDF for a dielectric function
    given in the format of Ashley (doi: 10.1016/0368-2048(88)80019-7).

    Ashley transforms the dielectric function ε(ω, q) to ε(ω, ω'); with an
    analytical relationship between ω and ω'. Therefore, we only need to store
    the probability distribution for ω'.

    This function computes the total mean free path and ICDF for ω', given a
    function dimfp(K, ω').

    This function calls dimfp() for ω' between 0 and K, despite conservation of
    momentum dictating that ω' < K/2. This is because some models (e.g. Kieft,
    doi: 10.1088/0022-3727/41/21/215310) ignore this restriction."""

    inel_imfp = np.zeros(K.shape) * units('nm^-1')
    inel_icdf = np.zeros((K.shape[0], P_omega.shape[0])) * units.eV

    for i, E in enumerate(K):
        tcs, icdf = compute_tcs_icdf(lambda w : dimfp(E, w), P_omega,
            np.linspace(0, E.magnitude, 100000)*E.units)
        inel_imfp[i] = tcs.to('nm^-1')
        inel_icdf[i] = icdf.to('eV')
        print('.', end='', flush=True)
    print()

    return inel_imfp, inel_icdf


def compile_full_imfp_icdf(elf_omega, elf_q, elf_data, # ELF data as function of (omega, q)
    K, P_omega, n_omega_q, P_q,                        # Energy and probability to evaluate at
    F):                                                # Fermi energy
    """Compute inverse mean free path, energy transfer ICDF and momentum transfer
    ICDF for an arbitrary dielectric function ε(ω, q).

    ε(ω, q) is given by elf_omega, elf_q and elf_data.

    The data is evaluated for energies given by K. P_omega is the probability
    axis for the energy loss ICDF.
    The momentum transfer ICDF is computed for each (K, omega), with the second
    parameter in n_omega_q evenly-spaced steps between 0 and K. The probability
    axis for the momentum ICDF is given by P_q.

    F is the Fermi energy of the material. This is used for a "Fermi correction"
    to prevent omega > K-F.

    This function is relativistically correct.
    """

    K_units = units.eV
    q_units = units('nm^-1')

    mc2 = units.m_e * units.c**2

    # Helper function, sqrt(2m/hbar^2 * K(1 + K/mc^2)), appears when getting
    # momentum boundaries from kinetic energy
    q_k = lambda _k : np.sqrt(2*units.m_e * _k*(1 + _k/(2 * mc2))) / units.hbar;

    def dcs(omega, q):
        # Linear interpolation, with extrapolation if out of bounds
        def find_index(a, v):
            low_i = np.clip(np.searchsorted(a, v, side='right')-1, 0, len(a)-2)
            vl = a[low_i]
            vh = a[low_i + 1]
            return low_i, (v - vl) / (vh - vl)
        low_x, frac_x = find_index(elf_omega.magnitude, omega.to(elf_omega.units).magnitude)
        low_y, frac_y = find_index(elf_q.magnitude, q.to(elf_q.units).magnitude)
        elf = (1-frac_x)*(1-frac_y) * elf_data[low_x, low_y] + \
			frac_x*(1-frac_y) * elf_data[low_x+1, low_y] + \
			(1-frac_x)*frac_y * elf_data[low_x, low_y+1] + \
			frac_x*frac_y * elf_data[low_x+1, low_y+1]

        elf[elf <= 0] = 0
        return elf / q

    eval_omega = np.geomspace(elf_omega[0].to(K_units).magnitude, \
        (K[-1]-F).to(K_units).magnitude, 10000) * K_units
    eval_q = np.geomspace(q_k(elf_omega[0]).to(q_units).magnitude, \
        2*q_k(K[-1]).to(q_units).magnitude, 10000) * q_units
    dcs_data = dcs(eval_omega[:,np.newaxis], eval_q)

    inel_imfp = np.zeros(K.shape) * units('nm^-1')
    inel_omega_icdf = np.zeros((K.shape[0], P_omega.shape[0])) * K_units
    inel_q_2dicdf = np.zeros((K.shape[0], n_omega_q, P_q.shape[0])) * q_units

    for i, E in enumerate(K):
        tcs, omega_icdf, q_2dicdf = compute_2d_tcs_icdf(dcs_data[eval_omega<E-F,:],
            eval_omega[eval_omega < E-F], eval_q,
            lambda omega : q_k(E) - q_k(np.maximum(0, E-omega)*K_units),
            lambda omega : q_k(E) + q_k(np.maximum(0, E-omega)*K_units),
            P_omega,
            np.linspace(0, (E-F).to(K_units).magnitude, n_omega_q)*K_units,
            P_q);
        tcs /= np.pi * units.a_0 * .5*(1 - 1 / (E/mc2 + 1)**2) * mc2

        inel_imfp[i] = tcs.to('nm^-1')
        inel_omega_icdf[i] = omega_icdf.to('eV')
        inel_q_2dicdf[i] = q_2dicdf.to('nm^-1')
        print('.', end='', flush=True)
    print()

    return inel_imfp, inel_omega_icdf, inel_q_2dicdf
