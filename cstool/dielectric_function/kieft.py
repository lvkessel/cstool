from cslib import (units, Settings)
import numpy as np

def L_Kieft(K, w0, F):
    # For sqrt & log calls, we have to strip the units. pint does not like "where".

    a = (w0 / K).magnitude
    s = np.sqrt(1 - 2*a, where = (a <= .5), out = np.zeros(a.shape))

    L1_range = (a > 0) * (a < .5) * (K-F > w0) * (K > F)
    L2_range = (a > 0) * (K-F > w0) * (K > F)

    # Calculate L1
    x1 = np.divide(2, a, where=L1_range, out=np.zeros(a.shape)) * (1 + s) - 1
    x2 = K - F - w0
    x3 = K - F + w0
    L1 = 1.5 * np.log((x1 * x2 / x3).magnitude, where = L1_range, out = np.zeros(a.shape))

    # Calculate L2
    L2 = -np.log(a, where = L2_range, out = np.zeros(a.shape))

    return np.maximum(0, (w0 < 50 * units.eV) * L1
                      + (w0 > 50 * units.eV) * L2)

def L_Theulings(K, w0, F):
    a = (w0 / K).magnitude
    b = (F / K).magnitude
    s = np.sqrt(1 - 2*a, where = (a <= .5), out = np.zeros(a.shape))

    L1_range = (a > 0) * (a < .5) * (a - s < 1 - 2*b)
    L2_range = (a > 0) * (a < 1 - b)

    # Calculate L1
    wm = (1 + a - s)/2
    wp = np.minimum((1 + a + s)/2, 1 - b)
    L1 = np.log((wp - a) * wm / (wp * (wm - a)), where = L1_range, out = np.zeros(a.shape))

    # Calculate L2
    L2 = -np.log(a, where = L2_range, out = np.zeros(a.shape))

    return np.maximum(0, (w0 < 50 * units.eV) * L1
                      + (w0 > 50 * units.eV) * L2)

def dimfp_kieft(K, w, s: Settings):
    """Compute differential inverse mean free paths from the model of Kieft
    (doi: 10.1088/0022-3727/41/21/215310). This model has been derived from
    Ashley's.

    This function is to be called with a scalar parameter K for kinetic energy
    and a numpy arraw w for ω'. Returns a numpy array, with one value for each
    ω'."""
    mc2 = units.m_e * units.c**2
    return s.elf_file(w) * L_Kieft(K, w, s.band_structure.fermi) \
            / (np.pi * units.a_0) \
            / (1 - 1 / (K/mc2 + 1)**2) / mc2

def dimfp_theulings(K, w, s: Settings):
    """Compute differential inverse mean free paths from the model of Theulings
    & van Kessel (unpublished). This model has been derived from Kieft's by
    correcting physically counterintuitive ingredients.

    L1 is a Fermi-corrected version of Ashley without the factor 3/2 rescale by
    Kieft; L2 is the same as in Kieft.

    This function is to be called with a scalar parameter K for kinetic energy
    and a numpy arraw w for ω'. Returns a numpy array, with one value for each
    ω'."""
    mc2 = units.m_e * units.c**2
    return s.elf_file(w) * L_Theulings(K, w, s.band_structure.fermi) \
            / (np.pi * units.a_0) \
            / (1 - 1 / (K/mc2 + 1)**2) / mc2
