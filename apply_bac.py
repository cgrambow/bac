#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
import json
import os

from bac import apply_bac
from util import read_csv, write_csv, read_xyz_file


def main():
    args = parse_args()
    cdata_list, _ = read_csv(args.calc_data)
    if args.geos is None:
        cdata = collections.OrderedDict(cdata_list)
        geos = None
    else:
        cdata, geos = collections.OrderedDict(), collections.OrderedDict()
        geos_list = read_xyz_file(args.geos)
        for (ident, h), geo in zip(cdata_list, geos_list):
            cdata[ident] = h
            geos[ident] = geo
    if args.expt_data is not None:
        edata_list, _ = read_csv(args.expt_data)
        edata = collections.OrderedDict(edata_list)
    else:
        edata = None
    with open(args.bacs) as f:
        bacs = json.load(f)

    cdata_new = apply_bac(cdata, bacs, geos=geos, data_comp=edata)
    name, ext = os.path.splitext(args.calc_data)
    path = name + '_bac' + ext
    write_csv(path, cdata_new)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('calc_data', help='CSV file with calculated enthalpies '
                                          '(SMILES/InChI in 1st column, enthalpy in 2nd column)')
    parser.add_argument('bacs', help='.json file containing BACs in kcal/mol')
    parser.add_argument('--expt_data', help='CSV file with experimental enthalpies (only for comparison) '
                                            '(SMILES/InChI in 1st column, enthalpy in 2nd column)')
    parser.add_argument('--geos', help='XYZ file with geometries in same order as calc_data. '
                                       'If not provided, BACs have to be based on atom/bond types')
    return parser.parse_args()


if __name__ == '__main__':
    main()
