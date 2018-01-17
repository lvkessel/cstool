from noodles import (schedule, schedule_hint)
from noodles.run.run_with_prov import run_parallel_opt
from noodles.display import NCDisplay
from cslib.noodles import registry

from cslib import Settings, Q_
from cslib.cs_table import DCS
from elsepa import elscata
import numpy as np


@schedule_hint(
    display="Running 'elscata' for Z={s.IZ},"
            " K=[{s.EV[0]:.2e~P}, ... ({s.EV.size})]  ",
    store=True,
    confirm=True,
    version="0.1.0")
def s_elscata(s):
    return elscata(s)


@schedule
def s_get_dcs(result, energies):
    def mangle_energy(e):
        s = 'dcs_{:.4e}'.format(e.to('eV').magnitude) \
            .replace('.', 'p').replace('+', '')
        return s[:9] + s[10:]

    dcs_keys = [mangle_energy(e) for e in energies]
    angles = np.cos(result[dcs_keys[0]]['THETA'])
    dcs = np.array([result[k]['DCS[0]'].to('cm²/sr').magnitude
                   for k in dcs_keys]) * Q_('cm²/sr')

    return DCS(energies[:, None], angles[::-1], dcs[:,::-1])


@schedule
def s_join_dcs(*dcs_lst):
    energy = np.concatenate(
        [dcs.energy for dcs in dcs_lst], axis=0) \
        * dcs_lst[0].energy.units
    angle = dcs_lst[0].q
    cs = np.concatenate(
        [dcs.cs for dcs in dcs_lst], axis=0) \
        * dcs_lst[0].cs.units
    return DCS(energy, angle, cs)


@schedule
def s_sum_dcs(material, **dcs):
    cs = material.rho_n * sum(element.count * dcs[symbol].cs
             for symbol, element in material.elements.items())
    first = next(iter(dcs))
    return DCS(dcs[first].energy, dcs[first].q, cs)


def mott_cs(material: Settings, energies, threads=4, mabs=False):
    """Compute Mott cross sections.

    Runs `elscata` for each element in the material. Then adds
    the cross-sections proportional to the composition of the
    material and returns differential inverse mean free paths
    as function of energy and cos(theta).

    :param material:
        Settings object containing parameters for this material.
    :param energies:
        Array quantity, 1-d array with dimension of energy.
    :param mabs:
        Enable absorbtion potential (computations can take longer).
    :param threads:
        Elsepa can take a long time to compute. This splits the
        `energies` array in several parts, and runs them in parallel.
    """
    def split_array(a, n):
        m = np.arange(0, a.size, a.size/n)[1:].astype(int)
        return np.split(a, m)

    chunks = split_array(energies, min(3*threads, len(energies)))

    def s_atomic_dcs(Z):
        no_muffin_Z = [1, 7, 8]

        settings = [Settings(
            IZ=Z, MNUCL=3, MELEC=4, IELEC=-1, MEXCH=1, IHEF=0,
            MCPOL=2, MABS=1 if mabs else 0,
            MUFFIN=0 if Z in no_muffin_Z else 1,
            EV=e) for e in chunks]

        f_results = [s_get_dcs(s_elscata(s), s.EV) for s in settings]

        return s_join_dcs(*f_results)

    dcs = {symbol: s_atomic_dcs(element.Z)
           for symbol, element in material.elements.items()}
    with NCDisplay() as display:
        mcs = run_parallel_opt(
            s_sum_dcs(material, **dcs),
            n_threads=threads, registry=registry,
            jobdb_file='cache.json', display=display)

    return mcs
