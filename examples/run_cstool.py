from cstool.parse_input import read_input, pprint_settings, cstool_model
from cstool.mott import mott_cs
from cstool.phonon import phonon_cs_fn
from cstool.dielectric_function import dimfp_kieft, elf_full_penn, \
    compile_full_imfp_icdf, compile_ashley_imfp_icdf
from cstool.compile import compute_tcs_icdf
from cstool.endf import compile_ionization_icdf
from cslib import units
from cslib.numeric import log_interpolate_f

from cslib.datafile import datafile

import numpy as np
import argparse


def compile_elastic(K, P, mott_fn):
    def elastic_cs_fn(E, costheta):
        return log_interpolate_f(
            lambda E: phonon_cs_fn(s)(E, costheta),
            lambda E: mott_fn(E, costheta),
            lambda x: x, 100*units.eV, 200*units.eV
        )(E)

    el_imfp = np.zeros(K.shape) * units('nm^-1')
    el_icdf = np.zeros((K.shape[0], P.shape[0])) * units.dimensionless

    for i, E in enumerate(K):
        tcs, icdf = compute_tcs_icdf(lambda costheta : elastic_cs_fn(E, costheta), \
            P, np.linspace(-1, 1, 100000))
        el_imfp[i] = 2*np.pi * tcs.to('nm^-1')
        el_icdf[i,:] = icdf
        print('.', end='', flush=True)
    print()

    return el_imfp, el_icdf




if __name__ == "__main__":
    #
    # Parse arguments
    #
    parser = argparse.ArgumentParser(
        description='Create HDF5 file from material definition.')
    parser.add_argument(
        'material_file', type=str,
        help="Filename of material in YAML format.")
    parser.add_argument(
        '--max_energy', type=float, default=50,
        help="Upper energy limit to use, in keV. Default is 50 keV.")
    args = parser.parse_args()
    max_energy = args.max_energy * units.keV
    
    
    #
    # Read parameters, print info
    #
    s = read_input(args.material_file)
    
    print(pprint_settings(cstool_model, s))
    print()
    print("Phonon loss: {:~P}".format(s.phonon.energy_loss))
    print("Total molar weight: {:~P}".format(s.M_tot))
    print("Number density: {:~P}".format(s.rho_n))
    print("Brillouin zone energy: {:~P}".format(s.phonon.E_BZ))
    print("Barrier energy: {:~P}".format(s.band_structure.barrier))
    print()
    
    
    
    #
    # Compute cross sections and write to file
    #
    with datafile("{}.mat.hdf5".format(s.name), 'w') as outfile:
        outfile.set_property('name', s.name)
        outfile.set_property('conductor_type', s.band_structure.model)
        outfile.set_property('fermi', s.band_structure.fermi, 'eV')
        outfile.set_property('barrier', s.band_structure.barrier, 'eV')
        outfile.set_property('phonon_loss', s.phonon.energy_loss, 'eV')
        outfile.set_property('density', s.rho_n, 'm^-3')
        outfile.set_property('effective_A', sum(e.M * e.count for e in s.elements.values())/(units.N_A*sum(e.count for e in s.elements.values())), 'g')
        if s.band_structure.model == 'insulator' or s.band_structure.model == 'semiconductor':
            outfile.set_property('band_gap', s.band_structure.band_gap, 'eV')
    
    
    
        # Elastic
        print("# Computing Mott cross-sections using ELSEPA.")
        e_mcs = np.logspace(1, 5, 145) * units.eV
        mcs = mott_cs(s, e_mcs, threads=4, mabs=False)
    
        print("# Computing elastic total cross-sections and iCDFs.")
        e_el = np.geomspace(.01, max_energy.to(units.eV).magnitude, 128) * units.eV
        p_el = np.linspace(0.0, 1.0, 1024)
        el_imfp, el_icdf = compile_elastic(e_el, p_el, mcs)
    
        elastic_grp = outfile.create_group("elastic")
        elastic_grp.add_scale("energy", e_el, 'eV')
        elastic_grp.add_dataset("imfp", el_imfp, ("energy",), 'nm^-1')
        elastic_grp.add_dataset("costheta_icdf", el_icdf, ("energy", None), '')
    
    
    
        # Inelastic-Kieft
        print("# Computing inelastic-Kieft total cross-sections and iCDFs.")
        e_inel = np.geomspace(s.band_structure.fermi.to(units.eV).magnitude+0.1, \
            max_energy.to(units.eV).magnitude, 128) * units.eV
        p_inel = np.linspace(0.0, 1.0, 1024)
        inel_imfp, inel_icdf = compile_ashley_imfp_icdf(
            lambda K,w : dimfp_kieft(K, w, s), e_inel, p_inel, s.band_structure.fermi)
    
        inelastic_grp = outfile.create_group("inelastic_kieft")
        inelastic_grp.add_scale("energy", e_inel, 'eV')
        inelastic_grp.add_dataset("imfp", inel_imfp, ("energy",), 'nm^-1')
        inelastic_grp.add_dataset("w0_icdf", inel_icdf, ("energy", None), 'eV')
    
    
    
        # Full Penn
        omega, q, elf = elf_full_penn(s.elf_file, max_energy, 1200, 1000)
    
        print("# Computing inelastic total cross-sections and iCDFs.")
        e_inel = np.geomspace(s.band_structure.fermi.to(units.eV).magnitude+0.1, \
            max_energy.to(units.eV).magnitude, 128) * units.eV
        p_inel = np.linspace(0.0, 1.0, 1024)
        inel_imfp, inel_omega_icdf, inel_q_2dicdf = compile_full_imfp_icdf(
            omega, q, elf,
            e_inel, p_inel, 1024, p_inel,
            s.band_structure.fermi)
    
        inelastic_grp = outfile.create_group("full_penn")
        inelastic_grp.add_scale("energy", e_inel, 'eV')
        inelastic_grp.add_dataset("imfp", inel_imfp, ("energy",), 'nm^-1')
        inelastic_grp.add_dataset("omega_icdf", inel_omega_icdf, ("energy", None), 'eV')
        inelastic_grp.add_dataset("q_icdf", inel_q_2dicdf, ("energy", None, None), 'nm^-1')
    
    
    
        # Ionization
        print("# Computing ionization energy probabilities")
        e_ion = np.geomspace(1, max_energy.to(units.eV).magnitude, 128) * units.eV
        e_frac = np.geomspace(1e-4, 1, 1024) * units.dimensionless
        p_ion = np.linspace(0.0, 1.0, 1024)
        ionization_icdf = compile_ionization_icdf(s, e_ion, e_frac, p_ion)

        ionization_grp = outfile.create_group("ionization")
        ionization_grp.add_scale("energy", e_ion, 'eV')
        ionization_grp.add_scale("loss_frac", e_frac, '')
        ionization_grp.add_dataset("binding_icdf", ionization_icdf, ("energy", "loss_frac", None), 'eV')

        ionization_grp.add_dataset("outer_shells", s.elf_file.get_outer_shells(), None, 'eV')
