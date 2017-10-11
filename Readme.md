How to use this code
====================
To train the DANN, you need a few changes to the Keras source files.
We recommend to make a new environment for this.
If the DANN is trained and you want to test it, we recommend to deactivate this new
environment, because sometimes there is a bug appearing and the predictions of the
NN are wrong.

INSTALL
-------
Create and activate a new environment:

::bash

  $ conda create -n dann anaconda
  $ source activate dann

Replace `training.py` in `/home/USERNAME/.conda/pkgs/keras-2.0.5-py36_0/lib/python3.6/site-packages/keras/engine`
by the `training.py` file in the folder. The file `training_old.py` is the original. Keep it, just in case.
If the path cannot be found, start Python and type:

::python

  import keras as ker
	ker.__file__

FILES
=====

GUTZWILLER
----------
Code from https://github.com/tcompa/BoseHubbardGutzwiller
Needs installing this files


Gradient Reverse Layer
----------------------
https://github.com/fchollet/keras/issues/3119#issuecomment-230289301

DANN Implementation
-------------------
https://github.com/fchollet/keras/pull/4031/files

Bogoliubuv_Kitaev
-----------------
Produces Kitaev states

SSH_states_and_Winding_Nr
-------------------------
Produces SSH states for OBC and PBC and calculates
also the winding number and gives a plot of it.

- To do the same for long range SSH, replace the
Hamiltonian in this file with the Hamiltonian from
`SSH_Long_Range_Hamiltonian.py`.
