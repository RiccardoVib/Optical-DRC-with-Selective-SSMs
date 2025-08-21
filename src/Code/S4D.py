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


import tensorflow as tf
from einops import repeat
import numpy as np


def selective_scan(u, dA, dB, dC, D, last_state, stateful):

    ub = u
    dB_u = tf.einsum('bld,bldn->bldn', ub, dB)

    dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])  # put zero at fist instant
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)  # 0, A, 2A ..
    dA_cumsum = tf.exp(dA_cumsum)  # 1, e^A, e^2A, .... -> 1, A, A^2 ...

    x = dB_u / (dA_cumsum + 1e-12)
    x = tf.math.cumsum(x, axis=1) * dA_cumsum

    if stateful:
        dA_cumsum_l = tf.math.cumsum(dA, axis=1)
        dA_cumsum_l = tf.exp(dA_cumsum_l)
        dA_cumsum_l *= tf.expand_dims(last_state, axis=1)
        x = x + dA_cumsum_l

    last_state = x[:, -1]
    y = tf.einsum('bldn,bln->bld', x, dC)

    return y + u * D, last_state


class S4D(tf.keras.layers.Layer):
    def __init__(self, model_states, model_input_dims, batch_size, mini_batch_size, stateful, hippo, dt_min=0.001,
                 dt_max=0.1):
        super(S4D, self).__init__()

        self.state = None
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.hippo = hippo
        self.model_input_dims = model_input_dims
        self.model_states = model_states
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.stateful = stateful

        log_A_real = tf.math.log(tf.constant(0.5 * tf.ones((self.model_input_dims, self.model_states))))
        A_imag = tf.constant(np.pi) * repeat(np.arange(model_states), 'n -> h n', h=self.model_input_dims)
        self.log_A_real = tf.Variable(log_A_real, name='log_A_real', trainable=True)
        self.A_imag = tf.Variable(A_imag, name='A_imag', trainable=True)

        B_real = tf.random.normal([self.model_input_dims, self.model_states], dtype=tf.float32)
        B_imag = tf.random.normal([self.model_input_dims, self.model_states], dtype=tf.float32)

        B = tf.concat([tf.expand_dims(B_real, axis=-1), tf.expand_dims(B_imag, axis=-1)], axis=-1)
        self.B = tf.Variable(B, name='B', trainable=True)

        log_dt = tf.random.uniform((1,)) * (tf.math.log(self.dt_max) - tf.math.log(self.dt_min)) + tf.math.log(
            self.dt_min)
        self.log_dt = tf.Variable(log_dt, trainable=True)

        C = tf.random.normal([1, self.model_states, 2], stddev=0.5 ** 0.5, dtype=tf.float32)
        self.C = tf.Variable(C, trainable=True)


        self.D = tf.Variable(
            np.ones(self.model_input_dims),
            trainable=True, dtype=tf.float32)

        # u = tf.Variable(tf.random.normal([self.batch_size, self.mini_batch_size, 1]), dtype='float32')
        self.reset_states()

    def reset_states(self):
        self.state = tf.Variable(
            tf.zeros((self.batch_size, self.model_input_dims, self.model_states), dtype=tf.float32), name='state',
            trainable=False)

    def call(self, u):
        last_state = self.state[:self.batch_size]
        res_state = self.state[self.batch_size:]

        y, y_state = self.ssm(u, last_state=last_state, stateful=self.stateful)

        if self.stateful:
            self.state.assign(tf.concat([y_state, res_state], axis=0))
        return y

    def ssm(self, u, last_state, stateful):

        Lambda = tf.cast(tf.complex(-tf.exp(self.log_A_real), self.A_imag), dtype=tf.complex64)

        C = tf.complex(self.C[..., 0], self.C[..., 1])
        B = tf.complex(self.B[..., 0], self.B[..., 1])

        step = tf.cast(tf.exp(self.log_dt), dtype=tf.complex64)

        dA = Lambda * step  # (H N)
        dB = B * (tf.exp(dA) - 1.) / Lambda

        dA = tf.tile(tf.reshape(dA, [1, 1, self.model_input_dims, self.model_states]),
                     [self.batch_size, self.mini_batch_size, 1, 1])
        dB = tf.tile(tf.reshape(dB, [1, 1, self.model_input_dims, self.model_states]),
                     [self.batch_size, self.mini_batch_size, 1, 1])
        C = tf.tile(tf.reshape(C, [1, 1, self.model_states]), [self.batch_size, self.mini_batch_size, 1])
        D = self.D

        y, y_state = selective_scan(u, dA, dB, C, D, last_state, stateful)

        return y, y_state