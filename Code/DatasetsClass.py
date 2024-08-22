import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from Utils import filterAudio
from scipy.signal.windows import tukey
import scipy.fft
import matplotlib.pyplot as plt

class DataGeneratorPicklesCL1B(Sequence):

    def __init__(self, data_dir, filename, input_size, n_pickles=1, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.data_dir = data_dir
        self.filename = filename
        self.count = 0
        self.batch_size = batch_size
        self.n_pickles = n_pickles
        self.window = input_size

        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)


        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][:, :], dtype=np.float32)
        x = x[:, :x.shape[1]//2]
        y = y[:, :y.shape[1]//2]
        if x.shape[0] == 1:
           x = np.repeat(x, y.shape[0], axis=0)
        x = x * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        z = np.array(Z['z'], dtype=np.float32)
        del Z

        rep = x.shape[1]
        x = x.reshape(-1)
        y = y.reshape(-1)

        N = int((x.shape[0] - self.window) / self.batch_size)-1 #how many iteration
        lim = int(N * self.batch_size) + self.window #how many samples
        x = x[:lim]
        y = y[:lim]
        z = np.repeat(z, rep, axis=0)

        return x, y, z, rep, lim

    def on_epoch_end(self):
        self.indices = np.arange(self.window, self.x.shape[0]+1)
        self.count = 0

    def reset_indices(self):
        self.indices = np.arange(self.window, self.x.shape[0]+1)

    def __len__(self):
        return int((self.x.shape[0]) / self.batch_size)*self.n_pickles

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = np.empty((self.batch_size, self.window))
        Y = np.empty((self.batch_size, 1))
        Z = np.empty((self.batch_size, 4))
        idx = idx - self.count*int((self.x.shape[0]) / self.batch_size)
        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0
        for t in range(indices[0], indices[-1]+1, 1):
            X[c, :] = np.array(self.x[t - self.window: t])
            Y[c, :] = np.array(self.y[t-1])
            Z[c, :] = np.array(self.z[t-1])
            c = c + 1

        Z1 = Z[:, :2]
        Z2 = Z[:, 2:]

        Xf = np.abs(scipy.fft.rfft(X, n=(8*32)-1))

        if indices[-1] == self.x.shape[0]-1 and self.n_pickles != 1:
            print('yes')
            self.reset_indices()
            self.count = self.count + 1
            del self.x, self.y, self.z
            self.x, self.y, self.z, _, _ = self.prepareXYZ(self.data_dir, self.filename[:-7] + str(self.count) + '.pickle')

        return [Z1, Z2, Xf, X], Y


class DataGeneratorPicklesLA2A(Sequence):

    def __init__(self, data_dir, filename, input_size, n_pickles=1, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.data_dir = data_dir
        self.filename = filename
        self.count = 0
        self.batch_size = batch_size
        self.n_pickles = n_pickles
        self.window = input_size

        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)


        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][:, :], dtype=np.float32)
        x = x[:, :x.shape[1]//2]
        y = y[:, :y.shape[1]//2]
        if x.shape[0] == 1:
           x = np.repeat(x, y.shape[0], axis=0)
     
        x = x * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        z = np.array(Z['z'], dtype=np.float32)
        del Z

        rep = x.shape[1]
        x = x.reshape(-1)
        y = y.reshape(-1)

        N = int((x.shape[0] - self.window) / self.batch_size)-1 #how many iteration
        lim = int(N * self.batch_size) + self.window #how many samples
        x = x[:lim]
        y = y[:lim]
        z = np.repeat(z, rep, axis=0)

        return x, y, z, rep, lim

    def on_epoch_end(self):
        self.indices = np.arange(self.window, self.x.shape[0]+1)
        self.count = 0

    def reset_indices(self):
        self.indices = np.arange(self.window, self.x.shape[0]+1)

    def __len__(self):
        return int((self.x.shape[0]) / self.batch_size)*self.n_pickles

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = np.empty((self.batch_size, self.window))
        Y = np.empty((self.batch_size, 1))
        Z = np.empty((self.batch_size, 2))
        idx = idx - self.count*int((self.x.shape[0]) / self.batch_size)

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0
        for t in range(indices[0], indices[-1]+1, 1):
            X[c, :] = np.array(self.x[t - self.window: t])
            Y[c, :] = np.array(self.y[t-1])
            Z[c, :] = np.array(self.z[t-1])
            c = c + 1

        Z1 = Z[:, :1]
        Z2 = Z[:, 1:]
        
        Xf = np.abs(scipy.fft.rfft(X, n=(8 * 32) - 1))

        return [Z1, Z2, Xf, X], Y

