# Copyright (C) 2023 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2025, "Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models" in Journal of Audio Engineering Society.


import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile


class MyLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Define the learning schedule
      :param initial_learning_rate: the initial learning rate [float]
      :param training_steps: the number of total training steps (iterations) [int]
    """
    def __init__(self, initial_learning_rate, training_steps):
        self.initial_learning_rate = initial_learning_rate
        self.steps = training_steps * 30

    def __call__(self, step):
        lr = tf.cast(self.initial_learning_rate * (0.25 ** (tf.cast(step / self.steps, dtype=tf.float32))),
                     dtype=tf.float32)
        return lr#tf.math.maximum(lr, 1e-6)

def writeResults(results, units, epochs, b_size, learning_rate, model_save_dir,
                 save_folder,
                 index):
    """
    write to a text the result and parameters of the training
      :param results: the results from the fit function [dictionary]
      :param units: the number of model's units [int]
      :param epochs: the number of epochs [int]
      :param b_size: the batch size [int]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param index: index for naming the file [string]

    """
    results = {
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss'],
        'units': units,
        'epochs': epochs
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(pred, inp, tar, model_save_dir, save_folder, fs, filename):
    """
    Plot the rendered results
      :param pred: the model's prediction  [array of floats]
      :param inp: the input [array of floats]
      :param tar: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param fs: the sampling rate [int]
      :param filename: the name of the file [string]
      """

    # the loop split the prediction in 5 second long extracts
    l = len(inp) // (fs * 5)

    for i in range(0, l - 1, 100):
        y = tar[i * fs * 5: (i + 1) * fs * 5]
        predictions = pred[i * fs * 5: (i + 1) * fs * 5]
        inp_ = inp[i * fs * 5: (i + 1) * fs * 5]
        time = np.arange(len(inp_))/fs
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, inp_, label='input', alpha=0.3)
        ax.plot(time, y, label='target', alpha=0.9)
        ax.plot(time, predictions, label='pred', alpha=0.7)
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/plot' + filename + str(i) + '.pdf', format='pdf')
        plt.close('all')


def plotTraining(loss_training, loss_val, model_save_dir, save_folder, name):
    """
    Plot the training against the validation losses
      :param loss_training: vector with training losses [array of floats]
      :param loss_val: vector with validation losses [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param fs: the sampling rate [int]
      :param filename: the name of the file [string]
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')
    plt.legend(loc='upper center')  # , bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/' + name + 'loss.png')
    plt.close('all')


def predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, fs, filename):
    """
    Render the prediction, target and input as wav audio file
      :param pred: the model's prediction  [array of floats]
      :param x_test: the input [array of floats]
      :param y_test: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param fs: the sampling rate [int]
      :param filename: the name of the file [string]
    """
    pred_name = filename + '_pred.wav'
    inp_name = filename + '_inp.wav'
    tar_name = filename + '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    predictions = (predictions.reshape(-1))
    x_test = x_test.reshape(-1)
    y_test = y_test.reshape(-1)

    wavfile.write(pred_dir, fs, predictions.reshape(-1))
    wavfile.write(inp_dir, fs, x_test.reshape(-1))
    wavfile.write(tar_dir, fs, y_test.reshape(-1))

    plotResult(predictions, x_test, y_test, model_save_dir, save_folder, fs, filename)


def checkpoints(model_save_dir, save_folder):
    """
    Define the path to the checkpoints saving the last and best epoch's weights
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
    """
    ckpt_path = os.path.normpath(
        os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
       os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1,
                                                       save_best_value=True)

    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss', mode='min', save_best_only=False, save_weights_only=True, verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest

