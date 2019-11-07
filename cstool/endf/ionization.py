from cslib import units, Settings

import numpy as np
import os
import json

from urllib.request import urlopen
from hashlib import sha1
from pkg_resources import resource_string, resource_filename


def obtain_endf_files():
    sources = json.loads(resource_string(__name__, '../data/endf_sources.json').decode("utf-8"))
    endf_dir = resource_filename(__name__, '../data/endf_data')
    os.makedirs(endf_dir, exist_ok=True)
    for name, source in sources.items():
        source['filename'] = '{}/{}.zip'.format(endf_dir, name)

        if os.path.isfile(source['filename']):
            with open(source['filename'], 'rb') as f:
                if sha1(f.read()).hexdigest() == source['sha1']:
                    print("using cached file {}".format(source['filename']))
                    continue
                else:
                    print("cached file {} has incorrect checksum".format(source['filename']))

        print("downloading {} file".format(name))
        try:
            with urlopen(source['url']) as response:
                data = response.read()
                if sha1(data).hexdigest() != source['sha1']:
                    raise Exception("downloaded file has incorrect checksum")
                with open(source['filename'], 'wb') as f:
                    f.write(data)
        except Exception as e:
            print("failed to download {} file ({})".format(name, e))
            exit()

    return sources


# loglog interpolation: zero if out-of-bounds on the left.
def interp_ll(x, xp, fp):
    OK = fp>0
    return np.exp(np.interp(
        np.log(x),
        np.log(xp[OK]),
        np.log(fp[OK]),
        left=-np.inf))

from .parse_endf import parse_electrons
from scipy.interpolate import griddata
# K: Kinetic energy (or energies) of interest, in eV
# omega: Energy loss(es) of interest, in eV
# Returns two lists:
#   Differential cross section dσ/dω in barn/eV for each shell
#   Binding energy for each shell
def get_dcs_loss(K, omega, Z):
    sources = obtain_endf_files()
    e_data = parse_electrons(sources['electrons']['filename'], Z)

    dcs = []
    binding = []

    for rx in e_data.reactions.values():
        if rx.MT <= 533:
            continue

        CS_shell = interp_ll(K, rx.cross_section.x, rx.cross_section.y)

        # Read cross section data
        primary_K = [p['E1'] for p in rx.products if p['ZAP'] == 11][0] # eV
        recoil_E = [[pep.flatten() for pep in p['Ep']]
                for p in rx.products if (p['ZAP'] == 11 and p['LAW'] == 1)][0] # eV
        recoil_P = [[pb.flatten() for pb in p['b']]
                for p in rx.products if (p['ZAP'] == 11 and p['LAW'] == 1)][0] # eV^-1

        # Make symmetry in recoil_E and recoil_P explicit
        for i in range(len(primary_K)):
            K = primary_K[i]
            re = recoil_E[i]
            rp = recoil_P[i]

            recoil_E[i] = np.r_[re,
                K-rx.binding_energy-re[-2::-1]]
            recoil_P[i] = np.r_[.5*rp,
                .5*rp[-2::-1]]

        # Obtain differential cross sections for desired K, omega
        # by log-log-log interpolation
        dcs.append(np.exp(griddata((
            np.log(np.repeat(primary_K, [len(a) for a in recoil_E])),
            np.log(np.concatenate(recoil_E) + rx.binding_energy)),
            np.log(np.concatenate([recoil_P[i] for i in range(len(primary_K))])),
            (np.log(K), np.log(omega)),
            fill_value = -np.inf
		))*CS_shell)

        binding.append(rx.binding_energy)

    return dcs, binding


# K: 1D np array with kinetic energies of interest
# omega_frac: 1D np array with fractional energy losses of interest
#             (as fraction of K, between 0 and 1)
# P: probabilities (between 0 and 1)
# Returns 3D array, shape (len(K), len(omega_frac), len(P))
def compile_ionization_icdf(s: Settings, K, omega_frac, P):
    # len(K) x len(omega_frac) array of interesting energy losses
    omega = K[:, np.newaxis] * omega_frac[np.newaxis, :]

    # Generate sorted list of shells, sorted by binding energy.
    # Each shell is represented by a dict:
    #   - B: binding energy
    #   - DIMFP (array, shape of omega): Differential inverse mean free path
    shells = []
    for element in s.elements.values():
        dcs_element, binding_element = get_dcs_loss(
            np.repeat(K.to(units.eV).magnitude, omega.shape[1]).reshape(omega.shape),
            omega.to(units.eV).magnitude, element.Z)

        for shelli in range(len(dcs_element)):
            shells.append({
                'B': binding_element[shelli] * units.eV,
                'DIMFP': dcs_element[shelli] * units.barn
                    * element.count * s.rho_n
            })
    shells.sort(key = lambda s : s['B'])

    # Compute the running cumulative differential inverse mean free paths
    # for each shell, and then normalize
    total_DIMFP = np.zeros(omega.shape) / units.nm
    for shell in shells:
        total_DIMFP += shell['DIMFP']
        shell['DIMFP_cum'] = np.copy(total_DIMFP)
    for shell in shells:
        shell['P_cum'] = np.divide(shell['DIMFP_cum'], total_DIMFP,
            out = np.zeros(shell['DIMFP_cum'].shape),
            where = total_DIMFP > 0/units.nm)

    # Build Inverse Cumulative Distribution Function
    icdf = np.zeros((len(K), len(omega_frac), len(P))) * units.eV
    icdf[:] = np.nan
    for shell in reversed(shells):
        icdf[np.logical_and(
                shell['DIMFP'][:,:,np.newaxis] > 0/units.nm,
                P[np.newaxis, np.newaxis, :] <= shell['P_cum'][:,:,np.newaxis]
            )] = shell['B']

    return icdf
