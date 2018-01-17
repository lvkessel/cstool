from cslib import (units, Settings)
import numpy as np

def L_Ashley_w_ex(K, w0, _):
    a = w0 / K
    return (1 - a) * np.log(4/a) - 7/4*a + a**(3/2) - 33/32*a**2


def L_Ashley_wo_ex(K, w0, _):
    a = w0 / K
    s = np.sqrt(1 - 2*a)
    return np.log((1 - a/2 + s)/(1 - a/2 - s))

def dimfp_ashley(K, w, s: Settings):
    """Compute differential inverse mean free paths from the model of Ashley
    (doi: 10.1016/0368-2048(88)80019-7). This model is not corrected for
    exchange.

    This function is to be called with a scalar parameter K for kinetic energy
    and a numpy arraw w for ω'. Returns a numpy array, with one value for each
    ω'."""
    return s.elf_file(w) * L_ashley_wo_ex(K, w, s.band_structure.fermi) \
            / (2 * K * np.pi * units.a_0)

def dimfp_ashley_exchange(K, w, s: Settings):
    """Compute differential inverse mean free paths from the model of Ashley
    (doi: 10.1016/0368-2048(88)80019-7). This model is corrected for exchange.

    This function is to be called with a scalar parameter K for kinetic energy
    and a numpy arraw w for ω'. Returns a numpy array, with one value for each
    ω'."""
    return s.elf_file(w) * L_ashley_w_ex(K, w, s.band_structure.fermi) \
            / (2 * K * np.pi * units.a_0)
