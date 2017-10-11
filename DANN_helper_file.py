#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
DANN helper file, with all necessary functions for
'Adversarial Domain Adaptation for Identifying Phase Transitions'.

We adapted a lot of code from the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin
Credits:
- Vanush Vaswani who made the Keras implementation of
  'Domain-Adversarial Training of Neural Networks' by Y. Ganin
  (https://github.com/fchollet/keras/pull/4031/files)
- Clayton Mellina (https://github.com/pumpikano/tf-dann) for providing
  a sketch of implementation (in TF) and utility functions.
- Yusuke Iwasawa
  (https://github.com/fchollet/keras/issues/3119#issuecomment-230289301)
  for Theano implementation (op) for gradient reversal.

Author:  Patrick Huembeli
'''
from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import pylab as plt
from Gradient_Reverse_Layer import GradientReversal
# from keras.engine.training import make_batches
# from keras.datasets import mnist_m
from keras import backend
backend.set_image_data_format('channels_first')


def train_loader(start, end, filename):
    x_batch = HDF5Matrix(filename, 'my_data', start=start, end=end)
    x_batch = np.stack([x_batch], axis=1)
    y_batch = HDF5Matrix(filename, 'my_labels', start=start, end=end)
    return x_batch, y_batch


class DANNBuilder(object):
    def __init__(self, nb_filters, nb_epoch, nb_classes, img_cols, img_rows,
                 nb_pool, nb_conv, batch_size):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD()
        self.nb_filters = nb_filters
        self.nb_epoch = nb_epoch
        self.nb_classes = nb_classes
        self.img_cols = img_cols
        self.img_rows = img_rows
        self.nb_pool = nb_pool
        self.nb_conv = nb_conv
        self.batch_size = batch_size

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Convolution2D(self.nb_filters, (self.nb_conv, self.nb_conv),
                            border_mode='valid',
                            activation='relu')(model_input)
        net = Convolution2D(self.nb_filters, (self.nb_conv, self.nb_conv),
                            activation='relu')(net)
        net = MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool))(net)
        net = Dropout(0.5)(net)
        net = Flatten()(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu')(model_input)
        net = Dropout(0.5)(net)
        net = Dense(self.nb_classes, activation='softmax',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=net)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, plot_model=False):
        # this is model until Flatten()
        net = self._build_feature_extractor(main_input)
        self.grl = GradientReversal(1.0)  # add GradientReversal
        branch = self.grl(net)
        # add feed forward part
        branch = Dense(128, activation='relu')(branch)
        branch = Dropout(0.1)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch)
        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        _TRAIN = K.variable(1, dtype='uint8')
        net = Lambda(lambda x: K.switch(K.learning_phase(),
                     x[:int(self.batch_size // 2), :], x),
                     output_shape=lambda x: ((self.batch_size // 2,) +
                     x[1:]) if _TRAIN else x[0:])(net)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=[branch, net])
        model.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(input=main_input,
                           output=self.domain_invariant_features)
        return tsne_model


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def batch_generator(len_x_train, id_array, batch_size, data, labels):
    np.random.shuffle(id_array)  # shuffling is fullfilled here
    for start in range(0, len_x_train, batch_size):
        end = min(start + batch_size, len_x_train)
        batch_ids = id_array[start:end]
        if labels is not None:
            x_batch = data[batch_ids]
            y_batch = labels[batch_ids]
            # this only works if labels is numpy array
            yield x_batch, y_batch
        else:
            x_batch = data[batch_ids]
            yield x_batch
