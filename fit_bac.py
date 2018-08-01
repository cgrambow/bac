#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
import os

import numpy as np

from bac import fit_bac


def main():
    args = parse_args()
    cdata_list, mults = read_csv(args.calc_data, third_col=args.with_mult, third_col_type=int)
    edata_list, uncertainties = read_csv(args.expt_data, third_col=args.weighted)

    if args.geo_exceptions is not None:
        with open(args.geo_exceptions) as f:
            geo_exceptions = {line.strip() for line in f if line.strip()}
    else:
        geo_exceptions = set()

    if args.geos is None:
        cdata = collections.OrderedDict(cdata_list)
        geos = None
    else:
        cdata, geos = collections.OrderedDict(), collections.OrderedDict()
        geos_list = read_xyz_file(args.geos)
        for (ident, h), geo in zip(cdata_list, geos_list):
            cdata[ident] = h
            geos[ident] = geo
    edata = collections.OrderedDict(edata_list)
    cdata_new = fit_bac(
        cdata, edata, args.out_dir, uncertainties=uncertainties,
        geos=geos, geo_exceptions=geo_exceptions, mults=mults,
        val_split=args.val_split, folds=args.folds, use_atom_features=args.atom_features,
        global_min=args.global_min, global_min_iter=args.global_min_iter, lam=args.lam)

    name, ext = os.path.splitext(os.path.basename(args.calc_data))
    path = os.path.join(args.out_dir, name + '_bac' + ext)
    write_csv(path, cdata_new)

    config_path = os.path.join(args.out_dir, 'bac_config.txt')
    with open(config_path, 'w') as f:
        configs = {'weighted regression': args.weighted,
                   'with molecular corrections': args.with_mult,
                   'global minimization': args.global_min,
                   'basin hopping iterations': args.global_min_iter if args.global_min else 1,
                   'regularization parameter': args.lam,
                   'validation split': args.val_split,
                   'number of folds': args.folds}
        f.write('Configuration used to fit BACs:\n')
        for k, v in configs.iteritems():
            f.write('{}:  {}\n'.format(k, v))


def read_csv(path, third_col=False, third_col_type=float):
    with open(path) as f:
        d = []
        d2 = {}
        for line in f:
            ls = line.strip().split()
            if ls:
                ident = ls[0]
                d.append((ident, float(ls[1])))
                if third_col:
                    d2[ident] = third_col_type(ls[2])
    return d, d2


def write_csv(path, data):
    with open(path, 'w') as f:
        for k, v in data.iteritems():
            f.write('{}   {}\n'.format(k, v))


def read_xyz_file(path):
    with open(path) as f:
        contents = [line.strip() for line in f]

    xyzs = []
    i = 0
    while i < len(contents):
        if not contents[i]:  # Empty line
            break
        else:
            natoms = int(contents[i])
            xyz = contents[(i+2):(i+2+natoms)]
            symbols, coords = [], []
            for line in xyz:
                data = line.split()
                symbols.append(data[0])
                coords.append([float(c) for c in data[1:]])
            xyzs.append((symbols, np.array(coords)))
            i += natoms + 2

    return xyzs


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('calc_data', help='CSV file with calculated enthalpies '
                                          '(SMILES/InChI in 1st column, enthalpy in 2nd column)')
    parser.add_argument('expt_data', help='CSV file with experimental enthalpies '
                                          '(SMILES/InChI in 1st column, enthalpy in 2nd column)')
    parser.add_argument('out_dir', help='Directory for output files')
    parser.add_argument('--weighted', action='store_true', help='Perform a weighted regression using uncertainties '
                                                                'in third column of expt_data')
    parser.add_argument('--val_split', type=float, default=0.0, help='Fraction to use as validation set')
    parser.add_argument('--folds', type=int, default=1, help='Number of folds for cross-validation '
                                                             '(overrides val_split if >1)')
    parser.add_argument('--atom_features', action='store_true', help='Use AACs instead of BACs')
    parser.add_argument('--geos', help='XYZ file with geometries in same order as calc_data. '
                                       'If not provided, use BACs based on atom/bond types')
    parser.add_argument('--geo_exceptions', help='Use the geometries for the identifiers in this file '
                                                 'even if they do not match')
    parser.add_argument('--with_mult', action='store_true', help='Use multiplicities supplied as third column '
                                                                 'in calc_data to calculate molecular corrections')
    parser.add_argument('--global_min', action='store_true', help='Use a global minimization algorithm')
    parser.add_argument('--global_min_iter', type=int, default=10, help='Basin hopping iterations')
    parser.add_argument('--lam', type=float, default=0.0, help='Regularization parameter')
    return parser.parse_args()


if __name__ == '__main__':
    main()
