import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from librosa import display
from scipy.io import wavfile
import librosa.display
from scipy import fft
from Utils import filterAudio


class MyLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, training_steps, epochs):
        self.initial_learning_rate = initial_learning_rate
        self.steps = training_steps * 30#epochs

    def __call__(self, step):
        lr = tf.cast(self.initial_learning_rate * (0.25 ** (tf.cast(step / self.steps, dtype=tf.float32))),
                     dtype=tf.float32)
        return lr#tf.math.maximum(lr, 1e-7)


def writeResults(results, units, epochs, b_size, learning_rate, model_save_dir,
                 save_folder,
                 index):
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


def plotResult(predictions, x, y, model_save_dir, save_folder, fs, filename):
    l = len(x) // (fs * 5)
    pred = predictions
    tar = y
    for i in range(0, l - 1, 100):
        y = tar[i * fs * 5: (i + 1) * fs * 5]
        predictions = pred[i * fs * 5: (i + 1) * fs * 5]
        inp = x[i * fs * 5: (i + 1) * fs * 5]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(inp, label='input', alpha=0.3)
        ax.plot(y, label='target', alpha=0.9)
        ax.plot(predictions, label='pred', alpha=0.7)
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' + save_folder + '/plot' + filename + str(i) + '.pdf', format='pdf')
        plt.close('all')


def plotTraining(loss_training, loss_val, model_save_dir, save_folder, name):
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
    pred_name = filename + '_pred.wav'
    inp_name = filename + '_inp.wav'
    tar_name = filename + '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    predictions = filterAudio(predictions.reshape(-1))
    predictions = (predictions.reshape(-1))
    x_test = x_test.reshape(-1)
    y_test = y_test.reshape(-1)

    wavfile.write(pred_dir, fs, predictions.reshape(-1))
    wavfile.write(inp_dir, fs, x_test.reshape(-1))
    wavfile.write(tar_dir, fs, y_test.reshape(-1))

    plotResult(predictions, x_test, y_test, model_save_dir, save_folder, fs, filename)


def checkpoints(model_save_dir, save_folder, initial_val):
    ckpt_path = os.path.normpath(
        os.path.join(model_save_dir, save_folder, 'Checkpoints', str(initial_val), 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', str(initial_val), 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
        os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1,
                                                       initial_value_thershold=initial_val, save_best_value=True)

    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss', mode='min', save_best_only=False, save_weights_only=True, verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest
