import tensorflow as tf
from scipy import signal
import numpy as np
import os
from keras import backend as K
import librosa

def RMSE(y_true, y_pred):
    """ root-mean-squared error """
    return K.mean(K.abs(K.sqrt(K.square(K.abs(y_pred))) - K.sqrt(K.square(K.abs(y_true)))))

def ESR(y_true, y_pred):
    """ error to signal ratio """
    return tf.divide(K.mean(K.square(y_pred - y_true)), K.mean(K.square(y_true) + 0.00001))


def MFCC(y_true, y_pred, sr):
    """ Mel-frequency cepstrum coefficients error """
    y_true_ = tf.reshape(y_true, [-1])
    y_pred_ = tf.reshape(y_pred, [-1])
    m = [1024]
    loss = 0
    for i in range(len(m)):

        pad_amount = int(m[i] // 2)  # Symmetric even padding like librosa.
        pads = [[pad_amount, pad_amount]]
        y_true = tf.pad(y_true_, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred_, pads, mode='CONSTANT', constant_values=0)

        Y_true = tf.signal.stft(y_true, frame_length=m[i], frame_step=m[i]//4,  fft_length=m[i], pad_end=False)
        Y_pred = tf.signal.stft(y_pred, frame_length=m[i], frame_step=m[i]//4,  fft_length=m[i], pad_end=False)

        Y_true = tf.abs(Y_true)
        Y_pred = tf.abs(Y_pred)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = Y_true.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 20000.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
        upper_edge_hertz)


        mel_spectrograms_pred = tf.tensordot(Y_pred, linear_to_mel_weight_matrix, 1)
        mel_spectrograms_pred.set_shape(Y_pred.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        mel_spectrograms_tar = tf.tensordot(Y_true, linear_to_mel_weight_matrix, 1)
        mel_spectrograms_tar.set_shape(Y_true.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))


        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms_pred = tf.math.log(mel_spectrograms_pred + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs_pred = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms_pred)

        log_mel_spectrograms_tar = tf.math.log(mel_spectrograms_tar + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs_tar = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms_tar)

        loss += tf.norm((mfccs_tar - mfccs_pred), ord=1) / (tf.norm(mfccs_tar, ord=1))
        loss = loss #/ y_true.shape[0]

    return loss

def flux(y_true, y_pred, sr):
    """ spectral flux error """

    y_t = []
    y_p = []
    w = 480000
    for i in range(0, len(y_true)-w, w//4):
        y_t.append(librosa.onset.onset_strength(y=y_true[i:i+w], sr=sr))
        y_p.append(librosa.onset.onset_strength(y=y_pred[i:i+w], sr=sr))
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    #y_true = y_t/y_t.max()
    #y_pred = y_p/y_p.max()

    return K.mean(tf.abs(y_t - y_p))

def STFT_t(y_true, y_pred):
    """ multi-STFT error """
    m = [32, 64, 128]
    y_true_ = tf.reshape(y_true, [-1])
    y_pred_ = tf.reshape(y_pred, [-1])

    loss = 0
    log_loss = 0
    for i in range(len(m)):

        pad_amount = int(m[i] // 2)  # Symmetric even padding like librosa.
        pads = [[pad_amount, pad_amount]]
        y_true = tf.pad(y_true_, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred_, pads, mode='CONSTANT', constant_values=0)

        Y_true = K.abs(tf.signal.stft(y_true, fft_length=m[i], frame_length=m[i], frame_step=m[i] // 4, pad_end=False))
        Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=m[i], frame_length=m[i], frame_step=m[i] // 4, pad_end=False))

        Y_true = tf.pow(K.abs(Y_true), 2)
        Y_pred = tf.pow(K.abs(Y_pred), 2)

        loss += tf.norm((Y_true - Y_pred), ord=1) / (tf.norm(Y_true, ord=1) + 0.00001)

    return (loss + log_loss) / len(m)
