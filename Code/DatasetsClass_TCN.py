import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey

class DataGeneratorPicklesCL1B(Sequence):

    def __init__(self, data_dir, filename, input_size, o, w, cond, batch_size=10):
        """
        Initializes a data generator object for the CL1B dataset
          :param filename: the name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_size: the input size
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.out_size = o
        self.input_size = input_size
        self.window = w
        self.cond = cond

        # prepare the input, target and conditioning matrix
        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)

        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        # load all the audio files
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)

        x = np.array(Z['x'][:, :], dtype=np.float32)
        lim = (x.shape[1] // self.input_size)*self.input_size
        x = x[:,:lim]
        y = np.array(Z['y'][:, :lim], dtype=np.float32)

        # if input is shared to all the targets, it is repeat accordingly to the number of target audio files
        if x.shape[0] == 1:
           x = np.repeat(x, y.shape[0], axis=0)

        # windowing the signal to avoid misalignments
        x = x * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        if self.cond:
            z = np.array(Z['z'][54:], dtype=np.float32)
        del Z

        # reshape to one dimension
        rep = x.shape[1]
        x = x.reshape(-1)
        y = y.reshape(-1)

        N = int((x.shape[0] - self.input_size) / self.batch_size)-1 #how many iteration
        lim = int(N * self.batch_size) + self.input_size #how many samples
        x = x[:lim]
        y = y[:lim]
        if self.cond:
            z = np.repeat(z, rep, axis=0)
            return x, y, z, rep, lim
        else:
            return x, y, 0., rep, lim
            
    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.input_size, self.x.shape[0]+1, self.out_size)


    def __len__(self):
        # compute the needed number of iteration before conclude one epoch
        return int((self.x.shape[0]) / self.batch_size / self.out_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # Initializing input, target, and conditioning batches
        X = np.empty((self.batch_size, self.input_size))
        Y = np.empty((self.batch_size, self.out_size))
        Z = np.empty((self.batch_size, 4))

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0
        if self.cond:
            for t in indices:
                X[c, :] = np.array(self.x[t - self.input_size: t])
                Y[c, :] = np.array(self.y[t - self.out_size: t])
                Z[c, :] = np.array(self.z[t-1])
                c = c + 1
            return [Z, X], Y
        else:
            for t in indices:
                X[c, :] = np.array(self.x[t - self.input_size: t])
                Y[c, :] = np.array(self.y[t - self.out_size: t])
                c = c + 1
            return X, Y
        


class DataGeneratorPicklesLA2A(Sequence):

    def __init__(self, data_dir, filename, input_size, o, w, cond, batch_size=10):
        """
        Initializes a data generator object for the CL1B dataset
          :param filename: the name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_size: the input size
          :param batch_size: The size of each batch returned by __getitem__
        """


        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.out_size = o
        self.input_size = input_size
        self.window = w
        self.cond = cond

        # prepare the input, target and conditioning matrix
        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)

        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)

        x = np.array(Z['x'][:, :], dtype=np.float32)
        lim = (x.shape[1] // self.input_size)*self.input_size
        x = x[:,:lim]
        y = np.array(Z['y'][:, :lim], dtype=np.float32)


        # if input is shared to all the targets, it is repeat accordingly to the number of target audio files
        if x.shape[0] == 1:
           x = np.repeat(x, y.shape[0], axis=0)

        # windowing the signal to avoid misalignments
        x = x * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        z = np.array(Z['z'], dtype=np.float32)
        del Z

        # reshape to one dimension
        rep = x.shape[1]
        x = x.reshape(-1)
        y = y.reshape(-1)

        N = int((x.shape[0] - self.input_size) / self.batch_size)-1 #how many iteration
        lim = int(N * self.batch_size) + self.input_size #how many samples
        x = x[:lim]
        y = y[:lim]
        z = np.repeat(z, rep, axis=0)

        return x, y, z, rep, lim

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.input_size, self.x.shape[0]+1, self.out_size)


    def __len__(self):
        # compute the needed number of iteration before conclude one epoch
        return int((self.x.shape[0]) / self.batch_size / self.out_size)


    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # Initializing input, target, and conditioning batches
        X = np.empty((self.batch_size, self.input_size))
        Y = np.empty((self.batch_size, self.out_size))
        Z = np.empty((self.batch_size, 2))

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0
        for t in indices:
            X[c, :] = np.array(self.x[t - self.input_size: t])
            Y[c, :] = np.array(self.y[t - self.out_size: t])
            Z[c, :] = np.array(self.z[t-1])
            c = c + 1
        return [Z, X], Y

