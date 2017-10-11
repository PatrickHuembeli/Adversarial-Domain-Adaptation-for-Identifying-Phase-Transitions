#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This is the Keras implementation of the DANN used in
'Adversarial Domain Adaptation for Identifying Phase Transitions'.

We adapted a lot of code from the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin
Credits:
- Vanush Vaswani who made the Keras implementation of
  'Domain-Adversarial Training of Neural Networks' by Y. Ganin
  (https://github.com/fchollet/keras/pull/4031/files)

Author:  Patrick Huembeli
'''
from __future__ import print_function
import tensorflow as tf
from keras.layers import Input
import keras.backend as K
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import pylab as plt

from sklearn.manifold import TSNE
from keras import backend
backend.set_image_data_format('channels_first')
from keras.models import load_model
from sklearn.cluster import KMeans
from DANN_helper_file import train_loader, DANNBuilder, plot_embedding, batch_generator


# ----------------------------------------------------------------
# Set Parameters
# -----------------------------------------------------------------
# DANN PARAMETERS
batch_size = 256
batch_size_test = 256
nb_epoch = 20
nb_classes = 2                  # number of classes
img_rows, img_cols = 64, 64     # input size --> double the system size!
nb_filters = 32                 # number of filters in Conv2D
nb_pool = 2                     # pool size of Conv2D
nb_conv = 3                     # Size of receptive field
_TRAIN = K.variable(1, dtype='uint8')
# -----------------------------------------------------------------
# WHAT DO YOU WANT TO RUN?
loading = True          # Loads Data
train_dann = True       # Trains the model
evaluate_new = True     # Makes model predictions
unsupervised = True     # Investigates feature space
# ----------------------------------------------------------------
# DATA SPECIFICATION
run = 0
boundary = 'OBC'
start = -3.0
end = 3.0
w2 = 0.2        # This is disorder that will be tested / trained
# ----------------------------------------------------------------
# For evaluation
w_trained = 0.2  # This is disorder that the weights have been trained on
epoch = 10       # load weights of this specific epoch
# ----------------------------------------------------------------
# Prepare Data
# We dont have to shuffle, we do it in train_generator
len_x_train = 20001
len_x_target = 20001

folder = 'OBC/'

train_weight_path = folder + 'weights_w2_' + str(w2)+'_TESTING' + \
                    str((start, end))
test_weight_path = folder + 'weights_w2_' + str(w_trained) + '_TESTING' + \
                    str((start, end)) + '.npy'
unsuperivised_weight_path = test_weight_path
pred_target_path = folder + 'predictions_on_target_w2_' + str(w2) + '_' + \
                    str((start, end))
pred_source_path = folder + 'predictions_on_train_w2_' + str(w2) + '_' + \
                    str((start, end))

if loading:
    path_source = folder + 'run_' + str(run) + '_SOURCE_ABS_' + boundary + \
                    '_' + str(len_x_train) + '_w2_0.0_' + str((start, end)) + \
                    '_N32.h5'
    path_target = folder + 'run_' + str(run) + '_TARGET_ABS_' + boundary + \
                    '_' + str(len_x_target) + '_w2_' + str(w2) + '_' + \
                  str((start, end)) + '_N32.h5'
    x_target, y_target = train_loader(0, len_x_target, path_target)
    x_train, y_train = train_loader(0, len_x_train, path_source)
    y_train = np.array(y_train)
    del y_target  # we dont need it for training
    source_index_arr = np.arange(x_train.shape[0])
    target_index_arr = np.arange(x_target.shape[0])
# ------------------------------------------------------------------
# Build models
main_input = Input(shape=(1, img_rows, img_cols), name='main_input')

builder = DANNBuilder(nb_filters, nb_epoch, nb_classes, img_cols,
                      img_rows, nb_pool, nb_conv, batch_size)
# These are separate models
dann_model = builder.build_dann_model(main_input)
# this is the whole DANN
dann_vis = builder.build_tsne_model(main_input)
# this is CNN until Flatten for feature extraction


# -----------------------------------------------------------
# Training
batches_per_epoch = len_x_train / batch_size
num_steps = nb_epoch * batches_per_epoch
j = 0
if train_dann:
    print('Training DANN model')
    for i in range(nb_epoch):
        src_gen = batch_generator(len_x_train, source_index_arr,
                                  batch_size // 2, x_train, y_train)
        target_gen = batch_generator(len_x_target, target_index_arr,
                                     batch_size // 2, x_target, None)

        losses = list()
        acc = list()
        print('Epoch ', i)

        # THIS LINES CAN BE USED TO INTERRUPT TRAINING AND GO ON FROM
        # A CERTAIN EPOCH
        # if i == 0:
            # weight = np.load(test_weight_path) # reload weights
            # dann_model.set_weights(weight)
        # else:
            # np.save(train_weight_path, dann_model.get_weights())

        np.save(train_weight_path + str(i), dann_model.get_weights())
        device = "gpu"
        with tf.device("/" + device + ":0"):
            for (xb, yb) in src_gen:
                # Update learning rate and gradient multiplier as described in
                # the paper.
                p = float(j) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p)**0.75
                builder.grl.l = l
                builder.opt.lr = lr
                if xb.shape[0] != batch_size // 2:
                    continue
                try:
                    xt = next(target_gen)
                except:
                    # Regeneration
                    target_gen = target_gen(len_x_target, target_index_arr,
                                            batch_size // 2, x_target, None)
                # Concatenate source and target batch
                domain_labels = np.vstack([np.tile([0, 1], [len(xb), 1]),
                                          np.tile([1., 0.], [len(xt), 1])])
                xb = np.vstack([xb, xt])
                print(j)
                metrics = dann_model.train_on_batch({'main_input': xb},
                                                    {'classifier_output': yb,
                                                    'domain_output':
                                                        domain_labels},
                                                    check_batch_dim=False)
                j += 1

# --------------------------------------------------------
# Make model predictions and plot output of Classifier
if evaluate_new:
    weight = np.load(test_weight_path)  # reload weights
    dann_model.set_weights(weight)
    dann_model.summary()
    print('Evaluating target samples on DANN model')
    train_phase = []
    target_phase = []
    out2 = dann_model.predict(x_target, verbose=1)
    out1 = dann_model.predict(x_train, verbose=1)
    np.save(pred_target_path, out2)
    np.save(pred_source_path, out1)
    for j in range(0, len(out1[1])):
            train_phase.append(out1[1][j][0])
            target_phase.append(out2[1][j][0])
    plt.clf()
    plt.plot(np.linspace(0, 1, len(train_phase)), train_phase)
    plt.savefig(folder+'train')
    plt.clf()
    plt.plot(np.linspace(0, 1, len(target_phase)), target_phase)
    plt.savefig(folder+'target')

# ----------------------------------------------------------
# Investigate Feature space with unsupervised methods
if unsupervised:
    # Created mixed dataset for TSNE visualization
    num_test = 1000
    np.random.shuffle(source_index_arr)
    np.random.shuffle(target_index_arr)
    x_train = x_train[source_index_arr]
    y_train = y_train[source_index_arr]
    x_target = x_target[source_index_arr]
    weight = np.load(test_weight_path)  # reload weights
    dann_model.set_weights(weight)
    # ----------------------------
    # Test look only at target data
    combined_test_imgs = x_target[:num_test]
    combined_test_labels = y_train[:num_test]
    dann_embedding = dann_vis.predict(combined_test_imgs)
    dann_tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    tsne = dann_tsne.fit_transform(dann_embedding)
    plot_embedding(tsne, combined_test_labels.argmax(1),
                   combined_test_labels.argmax(1), 'DANN')
    plt.savefig('DANN_tSNE_with_train_label')

    kmeans = KMeans(n_clusters=2, random_state=0).fit(dann_embedding)
    k_labels = kmeans.labels_
    k_centres = kmeans.cluster_centers_
    plot_embedding(tsne, k_labels,
                   combined_test_labels.argmax(1), 'DANN')
    plt.savefig('DANN_tSNE_with_k_mean_label')
