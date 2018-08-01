#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


def calc_rmse(ytrue, ypred):
    diff = ytrue - ypred
    return np.sqrt(np.dot(diff, diff) / len(ytrue))


def calc_mae(ytrue, ypred):
    return np.sum(np.abs(ytrue - ypred)) / len(ytrue)


def shuffle_arrays(*args):
    n = len(args[0])
    assert all(len(a) == n for a in args)
    rng = np.random.RandomState(seed=7)
    for a in args:
        state = rng.get_state()
        rng.shuffle(a)
        rng.set_state(state)


def split_arrays(split, *args):
    """split is the fraction retained in the first part"""
    n = len(args[0])
    idx = int(n*split)
    arrays = [(a[:idx], a[idx:]) for a in args]
    return tuple(a for pair in arrays for a in pair)


def make_folds(folds, *args):
    n = len(args[0])
    fold_size = int(np.ceil(float(n) / folds))
    return tuple([a[i:(i+fold_size)] for i in range(0, n, fold_size)] for a in args)


def concat_folds(fold_num, *args):
    selected = [folded_arrays[fold_num] for folded_arrays in args]
    others = [np.concatenate(folded_arrays[:fold_num] + folded_arrays[(fold_num+1):])
              for folded_arrays in args]
    return tuple(a for pair in zip(others, selected) for a in pair)
