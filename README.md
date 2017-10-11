This is the computational appendix for the following paper:

Patrick Huembeli, Alexandre Dauphin, Peter Wittek. [Adversarial Domain Adaptation for Identifying Phase Transitions](https://arxiv.org/abs/1710.xxxxx). *arXiv:1710.xxxxx*, 2017.

DOI for this Git repository 
[![DOI](https://zenodo.org/badge/105749405.svg)](https://zenodo.org/badge/latestdoi/105749405)

# Installation

To train the DANN, you need a few changes to the Keras source files.
We recommend to make a new environment for this. We assume that the Python distribution is Anaconda. Create and activate a new environment:

```bash
$ conda create -n dann anaconda
$ source activate dann
```
Replace `training.py` in `/home/USERNAME/.conda/pkgs/keras-2.0.5-py36_0/lib/python3.6/site-packages/keras/engine`
by the `training.py` file in the folder. The file `training_old.py` is the original. Keep it, just in case.
If the path cannot be found, start Python and type:

::python

  import keras as ker
	ker.__file__

# Files to Generate States

## Bose_Hubbard_Gutzwiller_coefficients.py

- Code partially from [https://github.com/tcompa/BoseHubbardGutzwiller](https://github.com/tcompa/BoseHubbardGutzwiller), install these files to use our code.
- With this file we generated the Gutzwiller coefficients.

## Bogoliubov_Kitaev.py
- Produces Kitaev states

## SSH_states_and_Winding_Nr.py
- Produces SSH states for OBC and PBC and calculates
also the winding number and gives a plot of it.

- To do the same for long range SSH, replace the
Hamiltonian in this file with the Hamiltonian from
`SSH_Long_Range_Hamiltonian.py`.

## Ising

- `Ising_energy_Gibbs_sampling.py` generates the Ising configurations via Monte Carlo method.
  The code has been made faster by using the beginning of each Markov chain more than once.
- `CNN_Ising.py` is a normal convolutional neural network, that can give the same output as the DANN.

# Files for the neural network

## Gradient_Reverse_Layer.py
- Code from [https://github.com/fchollet/keras/issues/3119#issuecomment-230289301](https://github.com/fchollet/keras/issues/3119#issuecomment-230289301).
- Needs to be in same folder as DANN main file `DANN_example.py`.

## DANN_helper_file.py
 - Summerizes all the important building blocks from [https://github.com/fchollet/keras/pull/4031/files](https://github.com/fchollet/keras/pull/4031/files).
 - Needs to be in same folder as DANN main file `DANN_example.py`.

## DANN_example.py
- Main file for the DANN, with all the parameters.
- First there is specified, which data has to be loaded.
Then there is a section for the training, one for the preddiction / evaluation of the DANN, and the last part is to apply unsupervised techniques on the feature space directly.


## training.py and training_old.py

- `training.py` is the new file that has to be placed in the keras backend.
- `training_old.py` is a backup of the original file.
