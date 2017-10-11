#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import qutip as qt
from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D)
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras import backend as K


def train_classifier(X_train, Y_train, img_channels, img_rows, img_cols,
                     regCNN, regularDense):
    # ----------------------------------------------------------------
    # Neural Network
    K.set_image_dim_ordering('th')  # defines order of input channel first
    model = Sequential()
    shape = (img_channels, img_rows, img_cols)
    model.add(Convolution2D(32, (3, 3), border_mode='same',
                            input_shape=shape,
                            kernel_regularizer=regularizers.l2(regCNN)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3),
                            kernel_regularizer=regularizers.l2(regCNN)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), border_mode='same',
                            kernel_regularizer=regularizers.l2(regCNN)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3),
                            kernel_regularizer=regularizers.l2(regCNN)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(regularDense)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_classes,
                    kernel_regularizer=regularizers.l2(regularDense)))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])
    csv_logger = CSVLogger('training.log')
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
    #                            patience=0, verbose=0, mode='auto')
    model.fit(X_train, Y_train,
              batch_size=128, epochs=nb_epoch, callbacks=[csv_logger])
    # score = model.evaluate(X_test, Y_test, verbose=1)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    model.save('models/model_no_field.h5')


# ----------------------------------------------------------------------------
def load_data(train, N, average, runs, Tcrit=0.0, h=0.0):
    X_train = []
    Y_train = []
    magn_list = []
    for t in temp:
        if train:
            filename = ('data_no_field/uniform_field_spin_hmax_' + str(h) +
                        'T_' + str(t) + 'N_' + str(N) +
                        '_number_of_config_per_temp_' +
                        str(average)+'runs_'+str(runs)+'_0')
        else:
            filename = ('data_field_h1_5/uniform_field_spin_hmax_' + str(h) +
                        'T_' + str(t) + 'N_' + str(N) +
                        '_number_of_config_per_temp_' + str(average) +
                        'runs_' + str(runs) + '_0')
        a = qt.qload(filename)
        magn = 0
        for i in a:
            magn += abs(np.sum(i))
            i = (i+1)/2  # renormalize
            X_train.append(i.reshape(1, 20, 20))
            if t < Tcrit:
                Y_train.append([1, 0])
            else:
                Y_train.append([0, 1])
        magn_list.append(magn/(N**2*average))
    X_train = np.array(X_train)
    return X_train, Y_train


if __name__ == '__main__':
    loss_function = 'binary_crossentropy'
    train = False

    nb_classes = 2  # number of classes to distinguish
    nb_epoch = 30  # number of epochs
    regCNN = 0.00  # regularization for CNN (does not work if >0)
    regularDense = 0.005  # 0.04 regularization for FFNN

    # input image dimensions
    img_rows, img_cols = 20, 20
    # the CIFAR10 images are RGB
    img_channels = 1
    N = 20
    average = 200
    runs = 400000
    temp = np.linspace(0.2, 5.0, 25)
    # temp = [0.5]

    if train:
        X_train, Y_train = load_data(train, N, average, runs,
                                     Tcrit=2.27, h=0.0)
        train_classifier(X_train, Y_train, img_channels, img_rows, img_cols,
                         regCNN, regularDense)
    else:
        X_train, Y_train = load_data(train, N, average, runs, Tcrit=1.2, h=1.5)
        model = load_model('models/model_trained_on_Ising_no_field_098_acc.h5')
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])
        scores = model.evaluate(X_train, Y_train, batch_size=32,
                                verbose=1, sample_weight=None)
        print(scores)
