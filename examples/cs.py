from noodles.run.run_with_prov import run_parallel_opt
from noodles.display import NCDisplay

from cstool.parse_input import read_input, pprint_settings, cstool_model
from cstool.mott import s_mott_cs
from cstool.phonon import phonon_cs_fn
from cstool.elf import read_elf_data
from cstool.inelastic import inelastic_cs_fn

from cslib.noodles import registry
from cslib import units
from cslib.dataframe import DCS
from cslib.numeric import log_interpolate

import numpy as np
from numpy import (log10)


def shift(dE):
    def decorator(cs_fn):
        def shifted(E, *args):
            return cs_fn(E + dE, *args)
        return shifted
    return decorator


if __name__ == "__main__":
    s = read_input("./data/materials/gold.json")

    print(pprint_settings(cstool_model, s))
    print()
    print("Phonon loss: {:~P}".format(s.phonon_loss))
    print("Total molar weight: {:~P}".format(s.M_tot))
    print("Number density: {:~P}".format(s.rho_n))
    print("Brioullon zone energy: {:~P}".format(s.E_BZ))
    print()
    print("Computing Mott cross-sections using ELSEPA.")

    e = np.logspace(1, 5, 145) * units.eV
    f_mcs = s_mott_cs(s, e, split=12, mabs=False)

    with NCDisplay() as display:
        mcs = run_parallel_opt(
            f_mcs, n_threads=4, registry=registry,
            jobdb_file='cache.json', display=display)

    mcs.save_gnuplot('{}_mott.bin'.format(s.name))

    print("Computing Phonon cross-sections.")
    e = np.logspace(-2, 3, 181) * units.eV
    pcs = DCS.from_function(phonon_cs_fn(s), e[:, None], mcs.angle)
    pcs.save_gnuplot('{}_phonon.bin'.format(s.name))

    print("Merging elastic scattering processes.")

    @shift(s.fermi)
    def elastic_cs_fn(E, a):
        return log_interpolate(
            lambda E: phonon_cs_fn(s)(E, a), lambda E: mcs(E, a),
            lambda x: x, 100*units.eV, 200*units.eV)(E)

    e = np.logspace(-2, 5, 129) * units.eV
    ecs = DCS.from_function(elastic_cs_fn, e[:, None], mcs.angle)
    ecs.save_gnuplot('{}_ecs.bin'.format(s.name))

    print("Reading inelastic scattering cross-sections.")
    elf_data = read_elf_data(s.elf_file)
    K_bounds = (s.fermi + 0.1 * units.eV, 1e4 * units.eV)
    K = np.logspace(
        log10(K_bounds[0].to('eV').magnitude),
        log10(K_bounds[1].to('eV').magnitude), 1024) * units.eV
    w = np.logspace(
        log10(elf_data['w0'][0].to('eV').magnitude),
        log10(K_bounds[1].to('eV').magnitude / 2), 1024) * units.eV
    ics = DCS.from_function(inelastic_cs_fn(s), K[:, None], w)
    ics.save_gnuplot('{}_ics.bin'.format(s.name))
