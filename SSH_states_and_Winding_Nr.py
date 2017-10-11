#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to create SSH states and calculate the winding numbers for certain system
size.

Created on Mon Jul 31 10:20:50 2017

@author: Alexandre Dauphin and Patrick Huembeli
"""

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg as la
import h5py


def ham(j1, j2, w1, w2, n, per='no'):
    # j1 is the intracell hopping,j2 is the intercell hopping
    # w1 is the intracell disorder,w2 is the intercell disorder
    # if per is 'yes', the system has periodic boundary conditions
    vec1 = j1*np.ones(2*n-1)
    vec1[1:2*n-1:2] = j2
    vecdis = (2*np.random.random(2*n-1)-1)/2.
    vecdis[0:2*n:2] = vecdis[0:2*n:2]*w1
    vecdis[1:2*n-1:2] = vecdis[1:2*n-1:2]*w2
    mat = np.diag(vec1, k=1) + np.diag(vec1, k=-1) + np.diag(vecdis, k=1) + \
        np.diag(vecdis, k=-1)
    if per == 'yes':
        mat[0, 2*n-1] = j2
        mat[2*n-1, 0] = j2
    return mat


def gs(j1, j2, w1, w2, n, per='no'):
    # diagonalizes the Hamiltonian and finds the ground state
    mat = ham(j1, j2, w1, w2, n, per)
    _, vec = la.eigh(mat)
    return abs(vec)


def winding(j1, j2, w1, w2, n):
    # Calculates winding number
    mat = ham(j1, j2, w1, w2, n)
    _, vec = la.eigh(mat)
    p = vec[:, 0:n-2]
    q = vec[:, n+2:]
    p1 = np.dot(p, np.conjugate(np.transpose(p)))
    q1 = np.dot(q, np.conjugate(np.transpose(q)))
    x = []
    for i in np.arange(n):
        x = np.concatenate((x, [i-(n-1)/2, i-(n-1)/2]))
    x = np.diag(x)
    qq = p1-q1
    gammap = np.zeros(2*n)
    gammam = np.zeros(2*n)
    gammap[0:2*n:2] = np.ones(n)
    gammam[1:2*n:2] = np.ones(n)
    gammap = np.diag(gammap)
    gammam = np.diag(gammam)
    qpm = np.dot(gammap, np.dot(qq, gammam))
    qmp = np.dot(gammam, np.dot(qq, gammap))
    mat1 = np.dot(x, qpm)-np.dot(qpm, x)
    mat1 = np.dot(qmp, mat1)
    nu = np.trace(mat1) / n
    return nu


def calc_states(j2, w1, w2, N, run, nr_of_states, start, end,
                folder, per='no'):
    j1_list = np.linspace(start, end, nr_of_states)
    start_index = np.where(j1_list >= -1.)[0][0]
    end_index = np.where(j1_list >= 1.)[0][0]
    labels = [[0, 1]]*(start_index) + [[1, 0]]*(end_index-start_index) + \
             [[0, 1]]*(nr_of_states-end_index)
    edge_states = [gs(j1, j2, w1, w2, N, per=per) for j1 in j1_list]
    x_train = edge_states
    y_train = labels

    boundary = 'OBC'
    if per == 'yes':
        boundary = 'PBC'
    if w1 == 0.0:
        targ_source = '_SOURCE'
    else:
        targ_source = '_TARGET'

    filename = folder + 'run_' + str(run) + targ_source + '_ABS_' + \
               boundary + '_' + str(nr_of_states) + '_w2_' + str(w2) + '_' + \
               str((start, end)) + '_N' + str(N) + '.h5'

    f = h5py.File(filename, 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('my_data', (len(labels), 2*N, 2*N), dtype='f')
    X_dset[:] = x_train
    # Creating dataset to store labels
    y_dset = f.create_dataset('my_labels', (len(labels), 2), dtype='i')
    y_dset[:] = y_train
    f.close()


"""
Be careful if comparing to https://arxiv.org/pdf/1311.5233.pdf
m is j1, t is j2, W1 in the paper is the disorder of t,
here w1 is the disorder to j1, therefore m_n = m + w1*\omega_n
if we want to swipe also m, we fix t = j_2 = 1
In the paper they fix W_2 = 2*W_1 = W, which means, the disorder of t is
half the disorder of m
for us this means w1 = 2*w2
"""
j2 = 1.
w2 = 2.0
w1 = 2*w2
N = 64
j3, j4, w3, w4 = 0, 0, 0, 0
j1 = 0.8
states = gs(j1, j2, w1, w2, N, per='no')
plt.pcolormesh(states)
plt.show()

for w2 in [0.0, 2.0, 0.2, 0.5, 1.0]:
    j2 = 1.
    w1 = 2*w2
    n = 32
    j3, j4, w3, w4 = 0, 0, 0, 0
    average = 1000
    av = 0
    run = 0
    nr_of_states = 20000
    start = -3.
    end = 3.
    folder = 'OBC_new_21_09/'
    calc_states(j2, w1, w2, n, run, nr_of_states, start, end, folder, per='no')
    for i in range(average):
        print(w2, i)
        plotter = [winding(j1, j2, w1, w2, n)
                   for j1 in np.linspace(-2, 2, 201)]
        av += np.array(plotter)
    np.save(folder + 'Winding_number_N32_W' + str(w2) + 'average_' +
            str(average), av / average)
    plt.clf()
    plt.plot(av / average)
    plt.savefig(folder+'Winding'+str(int(w2*10)))
