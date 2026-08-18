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

TUKEY_ALPHA = 5e-6


class _TCNGenerator(Sequence):
     """
        Initializes a data generator object for the CL1B dataset
          :param filename: the name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_size: the input size
          :param batch_size: The size of each batch returned by __getitem__
        """
    
    _condition_channels = 4
    _skip_condition_rows = 0

    def __init__(self, data_dir, filename, input_size, out_size, window, cond, batch_size=10):
        if input_size <= 0 or out_size <= 0 or window <= 0 or batch_size <= 0:
            raise ValueError("input_size, out_size, window, and batch_size must be positive.")

        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = int(batch_size)
        self.out_size = int(out_size)
        self.input_size = int(input_size)
        self.window = int(window)
        self.cond = bool(cond)

        self.x, self.y, self.z = self.prepareXYZ(data_dir, filename)
        self.training_steps = len(self._ends) // self.batch_size
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        path = os.path.join(data_dir, filename)
        with open(path, "rb") as file_data:
            data = pickle.load(file_data)

        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)

        x = np.pad(
            x,
            pad_width=((0, 0), (self.input_size, 0)),
            mode="constant",
        )
        
        y = np.pad(
            y,
            pad_width=((0, 0), (self.input_size, 0)),
            mode="constant",
        )

        n_examples, n_samples = x.shape
        usable = (n_samples // self.input_size) * self.input_size
        if usable < self.input_size:
            raise ValueError("The dataset is shorter than input_size.")

        x = x[:, :usable]
        y = y[:, :usable]
        window = np.asarray(tukey(usable, alpha=TUKEY_ALPHA), dtype=np.float32)
        x *= window[None, :]
        y *= window[None, :]
        z = np.asarray(data["z"], dtype=np.float32)
        if z.shape[0] != n_examples and z.shape[1] == n_examples:
            z = z.T
    
        self._rep = usable
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = np.repeat(z, usable, axis=0)

        n_samples_flat = x.size
        first_end = max(self.input_size, self.out_size)
        last_end = n_samples_flat
        self._ends = np.arange(first_end, last_end + 1, self.out_size)
        self._ends = self._ends[: (len(self._ends) // self.batch_size) * self.batch_size]

        if len(self._ends) == 0:
            raise ValueError("No complete TCN batch can be formed.")

        return x, y, z

    def on_epoch_end(self):
        self.indices = self._ends.copy()

    def __len__(self):
        return len(self._ends) // self.batch_size

    def __call__(self):
        for idx in range(len(self)):
            yield self[idx]
        self.on_epoch_end()

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(f"Batch index {idx} is out of range for length {len(self)}.")

        ends = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = np.stack(
            [self.x[t - self.input_size : t] for t in ends], axis=0
        ).astype(np.float32, copy=False)
        Y = np.stack(
            [self.y[t - self.out_size : t] for t in ends], axis=0
        ).astype(np.float32, copy=False)

        if self.cond:
            Z = np.stack([self.z[t - 1] for t in ends], axis=0)
            Z = Z.astype(np.float32, copy=False)
            return [Z, X], Y

        return X, Y


class DataGeneratorPicklesCL1B(_TCNGenerator):
    pass


class DataGeneratorPicklesLA2A(_TCNGenerator):
    _condition_channels = 2

