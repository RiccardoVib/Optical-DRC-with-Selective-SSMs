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

TUKEY_ALPHA = 5e-6
FFT_SIZE = 128

class DataGeneratorPickles(Sequence):
    _z_channels = 2

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
        n_examples, n_samples = self.x.shape
        if n_examples % self.batch_size:
            raise ValueError(
                f"Number of examples ({n_examples}) must be divisible by "
                f"batch_size ({self.batch_size})."
            )
        if n_samples <= self.window:
            raise ValueError(
                f"The data contain {n_samples} samples, but input_size is "
                f"{self.window}; at least one window is required."
            )

        self._time_steps = (n_samples - self.window) // self.mini_batch_size
        self._example_batches = n_examples // self.batch_size
        self.training_steps = self._time_steps * self._example_batches

        self._reset_iteration()
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        path = os.path.join(data_dir, filename)
        with open(path, "rb") as file_data:
            data = pickle.load(file_data)

        try:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            z = np.asarray(data["z"], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f"Missing required dataset key: {exc.args[0]}") from exc
            

        x = x * np.array(tukey(x.shape[1], alpha=TUKEY_ALPHA), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=TUKEY_ALPHA), dtype=np.float32).reshape(1, -1)

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        usable_samples = self.window + ((x.shape[1] - self.window) // self.mini_batch_size) * self.mini_batch_size
        if usable_samples <= self.window:
            raise ValueError("No complete mini-batch can be formed from the data.")

        x = x[:, :usable_samples]
        y = y[:, :usable_samples]
        if z.shape[0] < z.shape[1]:
            z = z.T
        z = np.repeat(z[:, None, :], usable_samples, axis=1)
        return x, y, z

    def _reset_model_states(self):
        """Reset every stateful layer that exposes a reset method."""
        for layer in self.model.layers:
            reset = getattr(layer, "reset_states", None)
    
            if callable(reset):
                reset()
                
    def on_epoch_end(self):
        self.indices = np.arange(self.window, self.x.shape[1])
        self.indices2 = np.arange(self.x.shape[0])
        self._reset_model_states()

    def __len__(self):
        return int(self.training_steps)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Batch index {idx} is out of range for dataset of length {len(self)}."
            )
    
        # Position within the temporal sequence.
        time_batch_idx = idx % self._time_steps
        
        # Stateful models process each example batch from the first time batch.
        if time_batch == 0:
            self._reset_model_states()
            
        # Position within the example batches.
        example_batch_idx = idx // self._time_steps
    
        time_start = time_batch_idx * self.mini_batch_size
        example_start = example_batch_idx * self.batch_size
    
        time_indices = self.indices[
            time_start : time_start + self.mini_batch_size
        ]
    
        example_indices = self.indices2[
            example_start : example_start + self.batch_size
        ]
    
        if len(time_indices) != self.mini_batch_size:
            raise RuntimeError("Incomplete temporal mini-batch.")
    
        if len(example_indices) != self.batch_size:
            raise RuntimeError("Incomplete example batch.")
    
        X = np.stack(
            [
                self.x[
                    example_indices,
                    t - self.window : t
                ]
                for t in time_indices
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    
        Y = np.stack(
            [
                self.y[example_indices, t - 1]
                for t in time_indices
            ],
            axis=1,
        )[..., None].astype(np.float32, copy=False)
    
        Z = np.stack(
            [
                self.z[example_indices, t - 1]
                for t in time_indices
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    
        Z1 = Z[:, :, :self._z_channels] #ratio and threshold
        Z2 = Z[:, :, self._z_channels:] #attach and release
    
        Xf = np.abs(
            scipy.fft.rfft(
                X,
                n=FFT_SIZE,
                axis=-1,
            )
        ).astype(np.float32, copy=False)
    
        return [Z1, Z2, Xf, X], Y

class DataGeneratorPicklesCL1B(_PickleSequence):
    """Generator for CL1B data; preserves the original four-input interface."""

    _z_channels = 2


class DataGeneratorPicklesLA2A(_PickleSequence):
    """Generator for LA2A data; preserves the original four-input interface."""

    _z_channels = 1
