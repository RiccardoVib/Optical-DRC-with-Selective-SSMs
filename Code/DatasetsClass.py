# Copyright (C) 2025 Riccardo Simionato, University of Oslo
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


import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey
import scipy.fft

class DataGeneratorPicklesCL1B(Sequence):

    def __init__(self, data_dir, filename, input_size, mini_batch_size=1, batch_size=9, set='train', model=None):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.window = input_size
        self.model = model
        self.set = set

        self.x, self.y, self.z = self.prepareXYZ(data_dir, filename)
        assert self.x.shape[0] % self.batch_size == 0

        self.idj = 0
        self.idx = -1

        self.max_1 = (self.x.shape[1] // self.mini_batch_size)
        self.max_2 = (self.x.shape[0] // self.batch_size)
        self.max = self.max_1 * self.max_2
        self.training_steps = self.max
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][: :], dtype=np.float32)

        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        z = np.array(Z['z'], dtype=np.float32)
        del Z

        rep = x.shape[1]

        N = int((x.shape[1] - self.window) / self.mini_batch_size) #how many iteration
        lim = int(N * self.mini_batch_size) #how many samples
        x = x[:, :lim]
        y = y[:, :lim]
        if z.shape[0] < z.shape[1]:
            z = z.T
        z = np.repeat(z[:,np.newaxis,:], rep, axis=1)

        return x, y, z

    def on_epoch_end(self):
        self.indices = np.arange(self.window, self.x.shape[1])
        self.indices2 = np.arange(-1, self.x.shape[0])
        self.idj = 0
        self.idx = -1
        self.model.layers[4].reset_states()
        self.model.layers[12].reset_states()
        self.model.layers[13].reset_states()

    def __len__(self):
        return int(self.max)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = np.zeros((self.batch_size, self.mini_batch_size, self.window))
        Y = np.zeros((self.batch_size, self.mini_batch_size, 1))
        Z = np.zeros((self.batch_size, self.mini_batch_size, 4))

        if idx % self.max_1 - 1 == 0:
            self.idj += 1
            self.idx = -1
            self.model.layers[4].reset_states()
            self.model.layers[12].reset_states()
            self.model.layers[13].reset_states()
        self.idx += 1

        # get the indices of the requested batch
        indices = self.indices[self.idx*self.mini_batch_size:(self.idx+1)*self.mini_batch_size]
        indices2 = self.indices2[self.idj*self.batch_size:(self.idj+1)*self.batch_size]
        c = 0

        for t in range(indices[0], indices[-1]+1, 1):
            X[:, c, :] = np.array(self.x[indices2, t - self.window: t])
            Y[:, c, :] = np.array(self.y[indices2, t-1:t])
            Z[:, c, :] = np.array(self.z[indices2, t-1])
            c = c + 1

        Z1 = Z[:, :, :2]
        Z2 = Z[:, :, 2:]

        Xf = np.abs(scipy.fft.rfft(X, n=(8*32)-1))

        return [Z1, Z2, Xf, X], Y

class DataGeneratorPicklesLA2A(Sequence):

    def __init__(self, data_dir, filename, input_size, mini_batch_size=1, batch_size=9, set='train', model=None):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """

        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.window = input_size
        self.model = model
        self.set = set

        self.x, self.y, self.z = self.prepareXYZ(data_dir, filename)
        assert self.x.shape[0] % self.batch_size == 0

        self.idj = 0
        self.idx = -1

        self.max_1 = (self.x.shape[1] // self.mini_batch_size)
        self.max_2 = (self.x.shape[0] // self.batch_size)
        self.max = self.max_1 * self.max_2
        self.training_steps = self.max
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][: :], dtype=np.float32)

        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        z = np.array(Z['z'], dtype=np.float32)
        del Z

        rep = x.shape[1]

        N = int((x.shape[1] - self.window) / self.mini_batch_size) #how many iteration
        lim = int(N * self.mini_batch_size) #how many samples
        x = x[:, :lim]
        y = y[:, :lim]
        if z.shape[0] < z.shape[1]:
            z = z.T
        z = np.repeat(z[:,np.newaxis,:], rep, axis=1)

        return x, y, z

    def on_epoch_end(self):
        self.indices = np.arange(self.window, self.x.shape[1])
        self.indices2 = np.arange(-1, self.x.shape[0])
        self.idj = 0
        self.idx = -1
        self.model.layers[4].reset_states()
        self.model.layers[12].reset_states()
        self.model.layers[13].reset_states()

    def __len__(self):
        return int(self.max)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = np.zeros((self.batch_size, self.mini_batch_size, self.window))
        Y = np.zeros((self.batch_size, self.mini_batch_size, 1))
        Z = np.zeros((self.batch_size, self.mini_batch_size, 4))

        if idx % self.max_1 - 1 == 0:
            self.idj += 1
            self.idx = -1
            self.model.layers[4].reset_states()
            self.model.layers[12].reset_states()
            self.model.layers[13].reset_states()
        self.idx += 1

        # get the indices of the requested batch
        indices = self.indices[self.idx*self.mini_batch_size:(self.idx+1)*self.mini_batch_size]
        indices2 = self.indices2[self.idj*self.batch_size:(self.idj+1)*self.batch_size]
        c = 0

        for t in range(indices[0], indices[-1]+1, 1):
            X[:, c, :] = np.array(self.x[indices2, t - self.window: t])
            Y[:, c, :] = np.array(self.y[indices2, t-1:t])
            Z[:, c, :] = np.array(self.z[indices2, t-1])
            c = c + 1

        Z1 = Z[:, :1]
        Z2 = Z[:, 1:]
        
        Xf = np.abs(scipy.fft.rfft(X, n=(8 * 32) - 1))

        return [Z1, Z2, Xf, X], Y

