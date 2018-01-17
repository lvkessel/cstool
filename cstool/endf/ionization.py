from .parse_endf import parse_zipfiles

from cslib import units, Settings
from cslib.numeric import loglog_interpolate

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


def _ionization_shells(Z):
    sources = obtain_endf_files()
    return parse_zipfiles(sources['atomic_relax']['filename'],
                          sources['electrons']['filename'],
                          int(Z))


# Cross sections are premultiplied by their element's abundance and
# sorted from outer (low binding energy) to inner (high binding).
def ionization_shells(s: Settings):
    shells = []
    for element_name, element in s.elements.items():
        data = _ionization_shells(element.Z)
        for n, shell in data.items():
            K, cs = list(map(list, zip(*shell.cs.data)))
            B = shell.energy
            K = np.array(K)*units.eV
            cs = np.array(cs)*units.barn
            cs *= element.count
            shells.append({'B': B, 'K': K, 'cs': cs})
    shells.sort(key = lambda s : s['B']);
    return shells


def compile_ionization_icdf(shells, K, p_ion):
    # For each shell, get ionization cross sections as function of K and store in
    # shell['cs_at_K']. Total cross section over all shells goes to tcstot_at_K.
    tcstot_at_K = np.zeros(K.shape) * units('m^2')
    for shell in shells:
        shell['cs_at_K'] = np.zeros(K.shape) * units('m^2')
        i_able = K > shell['B']
        j_able = (shell['K'] > shell['B']) & (shell['cs'] > 0*units('m^2'))
        shell['cs_at_K'][i_able] = loglog_interpolate(
            shell['K'][j_able], shell['cs'][j_able])(K[i_able]).to('m^2')
        tcstot_at_K += shell['cs_at_K']

    # shell['P_at_K'] is the ionization probability of the present shell as fraction of total.
    # shell['Pcum_at_K'] is the cumulative probability for this shell and others before it.
    Pcum_at_K = np.zeros(K.shape)
    for shell in shells:
        shell['P_at_K'] = np.zeros(K.shape)
        i_able = (tcstot_at_K > 0*units('m^2'))
        shell['P_at_K'][i_able] = shell['cs_at_K'][i_able]/tcstot_at_K[i_able]
        Pcum_at_K += shell['P_at_K']
        shell['Pcum_at_K'] = np.copy(Pcum_at_K)

    # Compute ICDF from Pcum_at_K.
    ionization_icdf = np.ndarray((K.shape[0], p_ion.shape[0]))*units.eV
    for j, P in enumerate(p_ion):
        icdf_at_P = np.ndarray(K.shape) * units.eV
        icdf_at_P[:] = np.nan
        for shell in reversed(shells):
            icdf_at_P[P <= shell['Pcum_at_K']] = shell['B']
        ionization_icdf[:, j] = icdf_at_P
    # Round-off error in Pcum_at_K may cause the last icdf_at_P to remain undefined
    nans = np.logical_not(np.isfinite(ionization_icdf[:,-1]))
    ionization_icdf[nans,-1] = ionization_icdf[nans,-2]

    return ionization_icdf
