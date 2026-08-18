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

_EPS = tf.constant(1e-12, dtype=tf.float32)

def selective_scan(u, dA, dB, dC, D, last_state=None, stateful=False):
    """Run a complex-valued diagonal SSM scan and return real output."""
    u = tf.cast(u, tf.complex64)
    dA = tf.cast(dA, tf.complex64)
    dB = tf.cast(dB, tf.complex64)
    dC = tf.cast(dC, tf.complex64)
    D = tf.cast(D, tf.complex64)

    dB_u = tf.einsum("bld,bldn->bldn", u, dB)
    padded_dA = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])  # put zero at fist instant
    cumulative_A = tf.math.cumsum(padded_dA, axis=1)  # 0, A, 2A ..
    cumulative_A_exp = tf.exp(cumulative_A)  # 1, e^A, e^2A, .... -> 1, A, A^2 ...
    states = tf.cumsum(dB_u / (cumulative_A_exp + tf.cast(_EPS, tf.complex64)), axis=1)
    states = states * cumulative_A_exp

    if stateful and last_state is not None:
        initial_A = tf.exp(tf.cumsum(dA, axis=1))
        states = states + initial_A * tf.cast(last_state[:, None], tf.complex64)

    final_state = states[:, -1]
    output = tf.einsum("bldn,bln->bld", states, dC)
    output = output + u * D
    return tf.math.real(output), final_state
    

class S4D(tf.keras.layers.Layer):
    def __init__(self, model_states, model_input_dims, batch_size, mini_batch_size, stateful, hippo, dt_min=0.001,
                 dt_max=0.1):
        super(S4D, self).__init__()

        self.model_states = int(model_states)
        self.model_input_dims = int(model_input_dims)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.stateful = bool(stateful)
        self.hippo = hippo
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self._state = None

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

        self.reset_states()

    def _ensure_state(self, batch_size):
        if batch_size is None:
            return None
        shape = (int(batch_size), self.model_input_dims, self.model_states)
        if self._state is None or self._state.shape != tf.TensorShape(shape):
            self._state = self.add_weight(
                name="state",
                shape=shape,
                initializer="zeros",
                trainable=False,
            )
        return self._state
           
    def reset_states(self):
        if self._state is not None:
            self._state.assign(tf.zeros_like(self._state))

    def call(self, u):
        
        Lambda = tf.cast(tf.complex(-tf.exp(self.log_A_real), self.A_imag), dtype=tf.complex64)

        C = tf.complex(self.C[..., 0], self.C[..., 1])
        B = tf.complex(self.B[..., 0], self.B[..., 1])
        step = tf.cast(tf.exp(self.log_dt), dtype=tf.complex64)
       
        dA = Lambda * step # (H N)
        dB = B * tf.math.expm1(dA) / Lambda

        batch = tf.shape(u)[0]
        length = tf.shape(u)[1]
        dA = tf.broadcast_to(
            dA[None, None, :, :],
            [batch, length, self.model_input_dims, self.model_states],
        )
        dB = tf.broadcast_to(
            dB[None, None, :, :],
            [batch, length, self.model_input_dims, self.model_states],
        )
        dC = tf.broadcast_to(
            C[None, None, :, :],
            [batch, length, self.model_states],
        )
        
        output, final_state = selective_scan(
            u,
            dA,
            dB,
            dC,
            self.D,
            state,
            self.stateful,
        )
        if self.stateful and state is not None:
            state.assign(final_state)
        return output
