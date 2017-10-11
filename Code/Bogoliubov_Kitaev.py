#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to generate Input Data for Kitaev model.

Credits:
Hamiltonian from https://topocondmat.org/w1_topointro/1D.html
Bogoliubov according to
http://iopscience.iop.org/article/10.1088/0953-8984/25/47/475304

Author: Patrick Huembeli
"""
import sys
import numpy as np
from scipy import linalg as la
import h5py
# For simplicity we set t = delta


def hamiltonian(n, t, mu, delta):
    diagonal = mu*np.ones(n)
    diagonal = np.diag(diagonal, k=0)

    hopping = t*np.ones(n-1)
    hopping = np.diag(hopping, k=1) + np.diag(hopping, k=-1)

    pairing = delta*np.ones(n-1)
    matr = np.diag(-pairing, k=1) + np.diag(pairing, k=-1)

    kitaev = np.bmat([[diagonal + hopping, np.transpose(matr)],
                      [matr, -(diagonal+hopping)]])
    return kitaev


def gs(n, t, mu, delta):
    # diagonalize the Hamiltonian and finds the ground state
    mat = hamiltonian(n, t, mu, delta)
    _, vec = la.eigh(mat)
    return abs(vec)


nr_of_states = 1001
t = 1.0
mu = -2.5*t
delta = t
n = 64  # number of sites
start = -4.0
end = 4.0

# -----------------------------------------------------------------------
mu_list = np.linspace(start, end, nr_of_states)
start_index = np.where(mu_list >= -2.)[0][0]
end_index = np.where(mu_list >= 2.)[0][0]
labels = [[1, 0]]*(start_index) + [[0, 1]]*(end_index-start_index) + \
         [[1, 0]]*(nr_of_states-end_index)

if len(labels) != len(mu_list):
    sys.exit('Length of labels not equal length of states')

states = [gs(n, t, mu, delta) for mu in mu_list]
filename = 'Kitaev_20001_bigger.h5'
f = h5py.File(filename, 'w')

X_dset = f.create_dataset('my_data', (len(labels), 2*n, 2*n), dtype='f')
X_dset[:] = states

y_dset = f.create_dataset('my_labels', (len(labels), 2), dtype='i')
y_dset[:] = labels
f.close()
