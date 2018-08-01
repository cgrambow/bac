#!/usr/bin/env python
# -*- coding:utf-8 -*-

import collections
import json
import os

import numpy as np
import scipy.optimize

from mol import str_to_mol, geo_to_mol, get_features, get_bac_correction
from util import calc_rmse, calc_mae, shuffle_arrays, split_arrays, make_folds, concat_folds


def fit_bac(cdata, edata, out_dir, uncertainties=None,
            geos=None, geo_exceptions=None, mults=None,
            val_split=0.0, folds=1, use_atom_features=False,
            global_min=False, global_min_iter=10, lam=0.0):
    """
    cdata: Dictionary of calculated data
    edata: Dictionary of experimental data
    out_dir: Output directory
    uncertainties: Dictionary of uncertainties
    geos: Use BAC form with atom/bond types if geometries are not provided
    geo_exceptions: Override the geometry check for these identifiers
    mults: Dictionary of multiplicities
    val_split: Fraction of data to use as validation set
    folds: Number of folds for cross-validation (overrides val_split if >1)
    use_atom_features: Use atom features instead of bond features
    global_min: Use the basin hopping algorithm for global minimization
    lam: Regularization parameter
    Return dictionary of new calculated data
    """
    bond_types = True if geos is None else False

    # Only use calculated molecules that are also in experimental ones
    ids, mols, hexpt, hcalc, weights = [], [], [], [], []
    for ident, h in edata.iteritems():
        if ident in cdata:
            ids.append(ident)
            if bond_types:
                mols.append(str_to_mol(ident))
            else:
                mol = geo_to_mol(geos[ident])
                if ident not in geo_exceptions:
                    mol_check = str_to_mol(ident, single_bonds=True)
                    if not mol_check.isIsomorphic(mol):
                        raise Exception('Geometry does not match identifier {}'.format(ident))
                if mults:
                    mol.multiplicity = mults[ident]
                mols.append(mol)
            hexpt.append(h)
            hcalc.append(cdata[ident])
            if uncertainties:
                weights.append(1.0 / uncertainties[ident]**2.0)
    hexpt = np.array(hexpt)
    hcalc = np.array(hcalc)
    weights = np.array(weights)
    rmse_prev = calc_rmse(hexpt, hcalc)
    mae_prev = calc_mae(hexpt, hcalc)

    # Display the 10 worst ones before fitting
    diff = hexpt - hcalc
    print('Worst ones before fitting:')
    large_diff = [(ids[i], abs(d), hexpt[i], hcalc[i]) for i, d in enumerate(diff)]
    large_diff.sort(key=lambda _x: _x[1], reverse=True)
    for ident, d, hexpti, hcalci in large_diff[:10][::-1]:
        print('{}  {: .2f}  {: .2f}  {: .2f}'.format(ident, d, hexpti, hcalci))
    print

    # Shuffle data and set up some arrays
    if folds > 1:
        shuffle_arrays(ids, mols, hexpt, hcalc)
    rmses, rmses_train, rmses_val = [], [], []
    maes, maes_train, maes_val = [], [], []
    hbacs, output_strs, param_dicts = [], [], []

    def _postprocess(_hexpt, _hbac,
                     _hexpt_train, _hbac_train,
                     _hexpt_val, _hbac_val,
                     _output_str, _param_dict):
        rmses.append(calc_rmse(_hexpt, _hbac))
        maes.append(calc_mae(_hexpt, _hbac))
        _rmse_train = calc_rmse(_hexpt_train, _hbac_train)
        rmses_train.append(_rmse_train)
        maes_train.append(calc_mae(_hexpt_train, _hbac_train))
        _rmse_val = calc_rmse(_hexpt_val, _hbac_val)
        rmses_val.append(_rmse_val)
        maes_val.append(calc_mae(_hexpt_val, _hbac_val))
        hbacs.append(_hbac)
        output_strs.append(_output_str)
        param_dicts.append(_param_dict)
        print('RMSE train/val: {:.2f}/{:.2f}'.format(_rmse_train, _rmse_val))
        print('Parameters:')
        print(output_str)

    if bond_types:
        # Get number of atoms or bonds of each type as features
        if use_atom_features:
            features = [get_features(mol, atom_features=True, bond_features=False) for mol in mols]
        else:
            features = [get_features(mol, atom_features=False, bond_features=True) for mol in mols]
        feature_keys = list({k for f in features for k in f})
        feature_keys.sort()
        x, nocc = make_feature_mat(features, feature_keys)
        for idx in np.where(nocc <= 1)[0][::-1]:  # Remove features if they only occur once
            del feature_keys[idx]
        x, nocc = make_feature_mat(features, feature_keys)

        data = (features, hexpt, hcalc, weights)
        folded_data = make_folds(folds, *data)
        for fold_num in range(folds):
            print('Fold {}'.format(fold_num+1))
            if folds > 1:
                split_data = concat_folds(fold_num, *folded_data)
            else:
                # Split off validation data
                split_data = split_arrays(1.0-val_split, *data)
            features_train, features_val, hexpt_train, hexpt_val, hcalc_train, hcalc_val, weights_train, _ = split_data

            y_train = hexpt_train - hcalc_train
            x_train, nocc = make_feature_mat(features_train, feature_keys)
            weight_mat = np.diag(weights_train) if np.size(weights_train) > 0 else np.eye(len(x_train))
            w, ypred = lin_reg(x_train, y_train, weight_mat)
            hbac_train = hcalc_train + ypred

            xval, _ = make_feature_mat(features_val, feature_keys)
            hbac = hcalc + np.dot(x, w)
            hbac_val = hcalc_val + np.dot(xval, w)
            output_str = ''
            for fk, wi, n in zip(feature_keys, w, nocc):
                output_str += '{:<5} {: .4f}   {}\n'.format(fk, wi, n)
            param_dict = {fk: wi for fk, wi in zip(feature_keys, w)}
            _postprocess(hexpt, hbac, hexpt_train, hbac_train, hexpt_val, hbac_val,
                         output_str, param_dict)
    else:
        # Technically, it's possible that some atom type is not present in mols_train, but that's very unlikely
        all_atom_symbols = list({atom.element.symbol for mol in mols for atom in mol.atoms})
        all_atom_symbols.sort()
        nelements = len(all_atom_symbols)
        low, high = -1e6, 1e6  # Arbitrarily large, just so that we can use bounds in global minimization
        if mults:
            w0 = np.zeros(3 * nelements + 1) + 1e-6  # Order is a, aii, b, k
            wmin = [low] * nelements + [0] * nelements + [low] * nelements + [low]
            wmax = [high] * (3 * nelements + 1)
        else:
            w0 = np.zeros(3 * nelements) + 1e-6  # Order is a, aii, b
            wmin = [low] * nelements + [0] * nelements + [low] * nelements
            wmax = [high] * 3 * nelements
        bounds = [(l, h) for l, h in zip(wmin, wmax)]

        data = (mols, hexpt, hcalc, weights)
        folded_data = make_folds(folds, *data)
        for fold_num in range(folds):
            print('Fold {}'.format(fold_num+1))
            if folds > 1:
                split_data = concat_folds(fold_num, *folded_data)
            else:
                # Split off validation data
                split_data = split_arrays(1.0-val_split, *data)
            mols_train, mols_val, hexpt_train, hexpt_val, hcalc_train, hcalc_val, weights_train, _ = split_data
            weight_mat = np.diag(weights_train) if np.size(weights_train) > 0 else np.eye(len(mols_train))

            minimizer_kwargs = dict(method='SLSQP',  # Gradient-free minimization is a lot faster
                                    args=(all_atom_symbols, mols_train, hexpt_train, hcalc_train, weight_mat, lam),
                                    bounds=bounds)
            if global_min:
                take_step = RandomDisplacementBounds(wmin, wmax)
                res = scipy.optimize.basinhopping(objfun, w0, niter=global_min_iter,
                                                  minimizer_kwargs=minimizer_kwargs,
                                                  take_step=take_step)
            else:
                res = scipy.optimize.minimize(objfun, w0, **minimizer_kwargs)
            w = res.x
            print(res.fun)
            hbac = get_hbac(w, all_atom_symbols, mols, hcalc)
            hbac_train = get_hbac(w, all_atom_symbols, mols_train, hcalc_train)
            hbac_val = get_hbac(w, all_atom_symbols, mols_val, hcalc_val)

            a, aii, b, k = get_params(w, all_atom_symbols)
            param_dict = {'a': a, 'aii': aii, 'b': b, 'k': k}
            output_str = 'Atom  A       B       Aii\n'
            for s, wi in a.iteritems():
                output_str += ' {:<3} {: .4f} {: .4f} {: .4f}\n'.format(s, wi, b[s], aii[s])
            output_str += 'K = {:.4f}\n'.format(k)
            _postprocess(hexpt, hbac, hexpt_train, hbac_train, hexpt_val, hbac_val, output_str, param_dict)

    # Display the 10 worst ones after fitting (averaged across models)
    hbac_mean = np.mean(hbacs, axis=0)
    diff_new = hexpt - hbac_mean
    print('Worst ones after fitting:')
    large_diff = [(ids[i], abs(d), hexpt[i], hbac_mean[i]) for i, d in enumerate(diff_new)]
    large_diff.sort(key=lambda _x: _x[1], reverse=True)
    for ident, d, hexpti, hbaci in large_diff[:10][::-1]:
        print('{}  {: .2f}  {: .2f}  {: .2f}'.format(ident, d, hexpti, hbaci))

    rmse = np.mean(rmses)
    mae = np.mean(maes)
    rmse_train = np.mean(rmses_train)
    mae_train = np.mean(maes_train)
    rmse_val = np.mean(rmses_val)
    mae_val = np.mean(maes_val)
    print('\nAverages:')
    print('RMSE train/val: {:.2f}/{:.2f}'.format(rmse_train, rmse_val))
    print('MAE train/val: {:.2f}/{:.2f}'.format(mae_train, mae_val))
    print('Total RMSE before/after fitting: {:.2f}/{:.2f}'.format(rmse_prev, rmse))
    print('Total MAE before/after fitting: {:.2f}/{:.2f}'.format(mae_prev, mae))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    error_path = os.path.join(out_dir, 'errors.txt')
    bac_path = os.path.join(out_dir, 'bacs.txt')
    json_path = os.path.join(out_dir, 'bacs.json')
    with open(error_path, 'w') as f:
        f.write('RMSE before/after: {:.2f}/{:.2f}\n'.format(rmse_prev, rmse))
        f.write('MAE before/after: {:.2f}/{:.2f}\n'.format(mae_prev, mae))
    with open(bac_path, 'w') as f:
        for output_str in output_strs:
            f.write(output_str + '\n')
    with open(json_path, 'w') as f:
        if len(param_dicts) > 1:
            json.dump(param_dicts, f, indent=4, separators=(',', ': '))
        else:
            json.dump(param_dicts[0], f, indent=4, separators=(',', ': '))

    return collections.OrderedDict(zip(ids, hbac_mean))


def lin_reg(x, y, weight_mat):
    w = np.linalg.solve(np.dot(x.T, np.dot(weight_mat, x)), np.dot(x.T, np.dot(weight_mat, y)))
    ypred = np.dot(x, w)
    return w, ypred


def make_feature_mat(features, keys):
    x = np.zeros((len(features), len(keys)))
    for idx, f in enumerate(features):
        flist = []
        for k in keys:
            try:
                flist.append(f[k])
            except KeyError:
                flist.append(0.0)
        x[idx] = np.array(flist)
    nocc = np.sum(x, axis=0).astype(np.int)
    return x, nocc


def objfun(w, keys, mols, hexpt, hcalc, weight_mat, lam=0.0):
    """
    w: parameters
    keys: list of keys corresponding to parameters (w is of size 3*len(keys),
          because there are 3 parameters per element type)
    mols: molecules with geometries
    hexpt: experimental data
    hcalc: calculated data
    weight_mat: weight matrix
    lam: regularization parameter
    Return L2 loss
    """
    hbac = get_hbac(w, keys, mols, hcalc)
    diff = hexpt - hbac
    return np.dot(diff, np.dot(weight_mat, diff)) / len(hexpt) + lam * np.dot(w, w)


def get_hbac(w, keys, mols, hcalc):
    a, aii, b, k = get_params(w, keys)
    hcorr = np.array([get_bac_correction(mol, a, aii, b, k) for mol in mols])
    return hcalc - hcorr  # Note the sign here


def get_params(w, keys):
    nelements = len(keys)
    a = dict(zip(keys, w[:nelements]))
    aii = dict(zip(keys, w[nelements:2*nelements]))
    b = dict(zip(keys, w[2*nelements:3*nelements]))
    try:
        k = w[3*nelements]
    except IndexError:
        k = 0.0
    return a, aii, b, k


class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew
