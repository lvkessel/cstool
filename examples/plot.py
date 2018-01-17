import numpy as np
from sys import exit
from cslib.datafile import datafile
import matplotlib.pyplot as plt
from cslib import units

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot mean free paths from HDF5 material file.')
    parser.add_argument('material_file', type=str,
        help="Filename of material in HDF5 format.")
    parser.add_argument('--elastic', action='store_true')
    parser.add_argument('--inelastic_kieft', action='store_true')
    parser.add_argument('--inelastic_penn', action='store_true')
    args = parser.parse_args()

    infile = datafile(args.material_file, 'r')

    properties = {}

    print('properties:')
    for key in infile.list_properties():
        value = infile.get_property(key)
        print('{: <16} {}'.format(key, value))
    print()

    if not (args.elastic or args.inelastic_kieft or args.inelastic_penn):
        print('Not plotting anything. Use the -h flag for usage.')
        exit()

    plt.figure()
    energy_units = 'eV'
    mfp_units = 'nm'
    if args.elastic:
        group = infile.get_group('elastic')
        energy = group.get_dataset('energy')
        imfp = group.get_dataset('imfp')
        plt.loglog(energy.to(energy_units), (1/imfp).to(mfp_units), label='Elastic')

    if args.inelastic_kieft:
        group = infile.get_group('inelastic_kieft')
        energy = group.get_dataset('energy')
        imfp = group.get_dataset('imfp')
        plt.loglog(energy.to(energy_units), (1/imfp).to(mfp_units), label='Inelastic (Kieft)')

    if args.inelastic_penn:
        group = infile.get_group('full_penn')
        energy = group.get_dataset('energy')
        imfp = group.get_dataset('imfp')
        plt.loglog(energy.to(energy_units), (1/imfp).to(mfp_units), label='Inelastic (Full Penn)')

    plt.legend()
    plt.title('Mean free paths')
    plt.ylabel('$\lambda$ (%s)' % mfp_units)
    plt.xlabel('$K$ (%s)' % energy_units)
    plt.show()
    infile.close()
