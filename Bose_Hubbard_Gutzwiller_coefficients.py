#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to produce Gutzwiller coefficients
Needs some installation from
https://github.com/tcompa/BoseHubbardGutzwiller

Credits:
- https://github.com/tcompa/BoseHubbardGutzwiller

Author: Patrick Huembeli
"""
import math
import numpy
# import matplotlib
# matplotlib.use('Agg')
import pylab as plt
from lib_gutzwiller_simulated_annealing import Gutzwiller
from lib_gutzwiller_simulated_annealing import SA_for_gutzwiller


# Physical parameters
z = 200       # number of neighbors
nmax = 20     # cutoff on the occupation number per site
U = 1.0       # on-site interaction coefficient
mu = 0.5     # chemical potential
V = 0.00     # nearest-neighbor interaction coefficient
P = 0.0      # induced-tunneling coefficient


make_2D_plot = False


# Simulated-annealing parameters
beta_min = 0.1
beta_max = 1e4
cooling_rate = 0.025  # 0.05
n_steps_per_T = 1000
quench_to_T_equal_to_0 = True
n_steps_at_T_equal_to_0 = 10000


def produce_states(run, start, end, steps, J):
    mu_list = numpy.linspace(start, end, steps)
    coeff_out = []
    labels = []
    for mu in mu_list:
        G = Gutzwiller(nmax=nmax, U=U, zJ=J, mu=mu, zV=z*V, zP=z*P)
        G = SA_for_gutzwiller(G, beta_min=beta_min, beta_max=beta_max,
                              cooling_rate=cooling_rate, n_steps_per_T=n_steps_per_T,
                              quench_to_T_equal_to_0=quench_to_T_equal_to_0,
                              n_steps_at_T_equal_to_0=n_steps_at_T_equal_to_0)
        coeff = [i for i in G.f_new]
        coeff_out.append(coeff)
        dens = G.compute_density()
        print('mu:', mu, 'density:', dens)
        if mu % 1 == 0:
            if abs(dens - mu) > 10**(-3):
                labels.append([1., 0.])
                print('SF-Phase')
            else:
                labels.append([0., 1.])
                print('Mott-Phase')
        else:
            if abs(dens - math.ceil(mu)) > 10**(-2):
                labels.append([1., 0.])
                print('SF-Phase')
            else:
                labels.append([0., 1.])
                print('Mott-Phase')
    numpy.save('J_'+str(J)+'labels_source'+str(run), labels)
    numpy.save('J_'+str(J)+'coeff_source'+str(run), coeff_out)

J_list = numpy.linspace(0.003, 0.3, 100)
start = 0.03
end = 3.0
steps = 100
run_list = numpy.linspace(1, 100, 100)
for run in run_list:
    for J in J_list:
        produce_states(int(run), start, end, steps, J)


if make_2D_plot:  # make 2D plot for a 50x50 grid
    J_list = numpy.linspace(0.0, 0.3, 51)
    mu_list = numpy.linspace(0.0, 3.00, 51)
    mu_list = mu_list.tolist()
    meas_J = []
    coeff_J = []
    dens_der_J = []
    for J in J_list:
        meas_mu = []
        coeff_mu = []
        density_list = []
        dens_der_mu = []
        for mu in mu_list:
            # Initialize Gutzwiller-class instance
            G = Gutzwiller(nmax=nmax, U=U, zJ=J, mu=mu, zV=z*V, zP=z*P)
            # Perform simulated-annealing optimization
            G = SA_for_gutzwiller(G, beta_min=beta_min, beta_max=beta_max,
                                  cooling_rate=cooling_rate,
                                  n_steps_per_T=n_steps_per_T,
                                  quench_to_T_equal_to_0=quench_to_T_equal_to_0,
                                  n_steps_at_T_equal_to_0=n_steps_at_T_equal_to_0)
            density = G.compute_density()
            density_list.append(density)
            if mu_list.index(mu) == 0:
                dens_derivative = 1.0
            else:
                dens_derivative = density-density_list[-2]
            coeff = [i for i in G.f_new]
            coeff_mu.append(coeff)
            dens_der_mu.append(dens_derivative)
            meas_mu.append([J, mu, G.energy, density, dens_derivative])
        meas_J.append(meas_mu)
        coeff_J.append(coeff_mu)
        dens_der_J.append(dens_der_mu)
    numpy.save('Gutzwiller_find_boundaries_params_2D_plot', meas_J)
    numpy.save('Gutzwiller_find_boundaries_coeffs_2D_plot', coeff_J)
    print('This is the screen with 2D plot')
    plt.clf()
    plt.pcolormesh(numpy.array(dens_der_J))
    plt.savefig('states_plot')
