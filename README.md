# Adversarial-Domain-Adaptation-for-Identifying-Phase-Transitions
Python code for our paper "Adversarial Domain Adaptation for Identifying Phase Transitions"

## How to use this code:

To train the DANN there are changes that have to be done in the keras source files.
We recommend to make a new environment for this.


## INSTALL

- create new environment: conda create -n dann anaconda

- activate the environment: source activate dann

- replace 'training.py' in /home/USERNAME/.conda/pkgs/keras-2.0.5-py36_0/lib/python3.6/site-packages/keras/engine
by the 'training.py' file in the folder. 
'training_old.py' is the original file. Keep it, just in case.
If the path cannot be found start python and type:
	import keras as ker
	ker.__file__


## Files to Generate States

### Bose_Hubbard_Gutzwiller_coefficients.py

- Code partially from https://github.com/tcompa/BoseHubbardGutzwiller, install these files to use our code.
- With this file we generated the Gutzwiller coefficients.

### Bogoliubov_Kitaev.py
Produces Kitaev states

### SSH_states_and_Winding_Nr.py
Produces SSH states for OBC and PBC and calculates
also the winding number and gives a plot of it.

To do the same for long range SSH, replace the
Hamiltonian in this file with the Hamiltonian from
SSH_Long_Range_Hamiltonian.py

### Ising

- Ising_energy_Gibbs_sampling.py generates the Ising configurations via Monte Carlo method.
  The code has been made faster by using the beginning of each Markov chain more than once.
- CNN_Ising.py is a normal convolutional neural network, that can give the same output as the DANN.

## Files for Neural Network

### Gradient Reverse Layer
- Code from https://github.com/fchollet/keras/issues/3119#issuecomment-230289301
- Has to be in the same folder as the python script with the DANN code.

### DANN_helper_file
 Summerizes all the important building blocks from https://github.com/fchollet/keras/pull/4031/files.

### DANN Implementation
Main file for the DANN, with all the parameters.
First there is specified, which data has to be loaded.
Then there is a section for the training, one for the preddiction / evaluation of the DANN, and the last part is to apply unsupervised techniques on the feature space directly.




