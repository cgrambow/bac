#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pybel

import rmgpy.molecule.molecule as molecule
import rmgpy.molecule.element as elements

bond_symbols = {
    1: '-',
    2: '=',
    3: '#',
}

atomic_symbol_dict = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'S': 16,
}

atom_spins = {
    'H': 0.5,
    'C': 1.0,
    'N': 1.5,
    'O': 1.0,
    'S': 1.0,
}


def str_to_mol(identifier, single_bonds=False):
    if identifier.startswith('InChI'):
        mol = molecule.Molecule().fromInChI(identifier, backend='rdkit-first')
    else:
        mol = molecule.Molecule().fromSMILES(identifier)
    if single_bonds:
        return mol.toSingleBonds()
    else:
        return mol


def geo_to_mol(geo):
    """
    Use Open Babel for most cases because it's better at recognizing long bonds.
    Use RMG for hydrogen because Open Babel can't do it for mysterious reasons.
    """
    symbols, coords = geo
    if len(symbols) == 2 and all(s == 'H' for s in symbols):
        nums = np.array([atomic_symbol_dict[s] for s in symbols])
        mol = molecule.Molecule()
        mol.fromXYZ(nums, coords)
    else:
        xyz = '{}\n\n'.format(len(symbols))
        coords = ['{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}'.format(s, c) for s, c in zip(*geo)]
        xyz += '\n'.join(coords)
        mol = pybel.readstring('xyz', xyz)
        mol = pybel_to_rmg(mol)
    return mol


def pybel_to_rmg(pybelmol, addh=False):
    """ignore charge, multiplicity, and bond orders"""
    mol = molecule.Molecule()
    if addh:
        pybelmol.addh()
    for pybelatom in pybelmol:
        num = pybelatom.atomicnum
        element = elements.getElement(num)
        atom = molecule.Atom(element=element, coords=np.array(pybelatom.coords))
        mol.vertices.append(atom)
    for obbond in pybel.ob.OBMolBondIter(pybelmol.OBMol):
        begin_idx = obbond.GetBeginAtomIdx() - 1
        end_idx = obbond.GetEndAtomIdx() - 1
        bond = molecule.Bond(mol.vertices[begin_idx], mol.vertices[end_idx])
        mol.addBond(bond)
    return mol.toSingleBonds()


def get_features(mol, atom_features=False, bond_features=True):
    """
    Given a mol, extract the number of atoms of each type
    and/or the number of bonds of each type.
    """
    features = {}
    if atom_features:
        for atom in mol.atoms:
            symbol = atom.element.symbol
            features[symbol] = features.get(symbol, 0) + 1
    if bond_features:
        for bond in mol.getAllEdges():
            atom_symbol1 = bond.atom1.element.symbol
            atom_symbol2 = bond.atom2.element.symbol
            atom_symbols = [atom_symbol1, atom_symbol2]
            atom_symbols.sort()  # Make sure that representation is invariant to order of atoms in bond
            symbol = atom_symbols[0] + bond_symbols[bond.order] + atom_symbols[1]
            features[symbol] = features.get(symbol, 0) + 1
    return features


def get_bac_correction(mol, a, aii, b, k):
    """
    Given a mol with coordinates and dictionaries of BAC parameters,
    return the total BAC correction.
    """
    alpha = 3.0  # Angstrom^-1
    spin = 0.5 * (mol.multiplicity - 1)
    bac_mol = k * (spin - sum(atom_spins[atom.element.symbol] for atom in mol.atoms))
    bac_atom = sum(a[atom.element.symbol] for atom in mol.atoms)
    bac_bond = 0.0
    for bond in mol.getAllEdges():
        atom1 = bond.atom1
        atom2 = bond.atom2
        symbol1 = bond.atom1.element.symbol
        symbol2 = bond.atom2.element.symbol
        aab = (aii[symbol1] * aii[symbol2]) ** 0.5
        rab = np.linalg.norm(atom1.coords-atom2.coords)
        bca = bdb = 0.0
        for other_atom, other_bond in mol.getBonds(atom1).iteritems():
            if other_bond is not bond:
                other_symbol = other_atom.element.symbol
                bca += b[symbol1] + b[other_symbol]
        for other_atom, other_bond in mol.getBonds(atom2).iteritems():
            if other_bond is not bond:
                other_symbol = other_atom.element.symbol
                bdb += b[symbol2] + b[other_symbol]
        bac_bond += aab*np.exp(-alpha*rab) + bca + bdb
    return bac_mol + bac_atom + bac_bond
