#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long range SSH Hamiltonian

Created on Mon Jul 31 10:20:50 2017

@author: Alexandre Dauphin
"""
import numpy as np


def ham(j1, j2, j3, j4, w1, w2, w3, w4, n):
    vec1 = j1*np.ones(2*n-1)
    vec1[1:2*n-1:2] = j2
    vec2 = np.ones(2*n-3)*j3
    vec2[1:2*n-1:2] = j4
    vecdis = (2*np.random.random(2*n-1)-1)/2.
    vecdis[0:2*n:2] = vecdis[0:2*n:2]*w1
    vecdis[1:2*n-1:2] = vecdis[1:2*n-1:2]*w2
    vecdislr = (2*np.random.random(2*n-3)-1)/2.
    vecdislr[0:2*n-3:2] = vecdislr[0:2*n-3:2]*w3
    vecdislr[1:2*n-3:2] = vecdislr[1:2*n-3:2]*w4
    mat = np.diag(vec1, k=1) + np.diag(vec1, k=-1) + np.diag(vecdis, k=1) + \
          np.diag(vecdis, k=-1) + np.diag(vec2, k=3) + np.diag(vec2, k=-3) + \
          np.diag(vecdislr, k=3) + np.diag(vecdislr, k=-3)
    return mat
