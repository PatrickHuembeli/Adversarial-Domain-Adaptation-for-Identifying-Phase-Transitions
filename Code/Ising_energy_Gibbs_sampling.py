#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo sampling for Ising configurations.
Authors: Patrick Huembeli and Peter Wittek
"""

import matplotlib.pyplot as plt
import numpy as np


def get_energy(J, spins):
    return -J*np.sum(spins * np.roll(spins, 1, axis=0) +
                     spins * np.roll(spins, -1, axis=0) +
                     spins * np.roll(spins, 1, axis=1) +
                     spins * np.roll(spins, -1, axis=1))/2


def deltaE(J, spins, i, j):
    flip = -spins[i, j]
    delta = -2*J*flip*(spins[(i+1) % N, j] + spins[(i-1) % N, j] +
                       spins[i, (j+1) % N] + spins[i, (j-1) % N])
    return delta


def do_gibbs_sampling(interaction, spins, energy, temperature, n_samples):
    for _ in range(n_samples):
        i = np.random.randint(spins.shape[0])
        j = np.random.randint(spins.shape[1])
        delta = deltaE(interaction, spins, i, j)
        if delta < 0 or np.exp(-delta / temperature) > np.random.random():
            spins[i, j] *= -1
            energy += delta
    return spins, energy


def plot_probabilities(energies, T, bins=10):
    probabilities = np.exp(-np.array(sorted(energies))/T)
    Z = probabilities.sum()
    probabilities /= Z
    plt.plot(sorted(energies), probabilities)
    plt.ylim(0, 1.2*probabilities.max())
    plt.show()


def get_energy_distribution(N, temperature, interaction, n_runs, burnin_time,
                            n_samples, n_sample_distance):
    energy_list = []
    for run in range(n_runs):
        print("Run %d" % run)
        spins = np.random.choice([-1, 1], size=(N, N))
        energy = get_energy(interaction, spins)
        spins, energy = do_gibbs_sampling(interaction, spins, energy,
                                          temperature, burnin_time)
        for _ in range(n_samples):
            spins, energy = do_gibbs_sampling(interaction, spins, energy,
                                              temperature, n_sample_distance)
            energy_list.append(energy)
    return energy_list


temperature = 5
N = 5
interaction = 1.0
burnin_time = 10000
n_sample_distance = 1000
n_samples = 100
n_runs = 50

energies = get_energy_distribution(N, temperature, interaction, n_runs,
                                   burnin_time, n_samples, n_sample_distance)
plot_probabilities(energies, temperature, bins=50)
