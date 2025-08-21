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


import tensorflow as tf
from einops import rearrange, repeat
import numpy as np
import math

#### The code is adapted from https://github.com/state-spaces/mamba & https://github.com/PeaBrane/mamba-tiny


def selective_scan(u, delta, A, B, C, D, last_state, stateful, L):
    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
    dA_cumsum = tf.exp(dA_cumsum)

    x = dB_u / (dA_cumsum + 1e-12)
    x = tf.math.cumsum(x, axis=1) * dA_cumsum

    if stateful:
        dA_cumsum_l = tf.math.cumsum(dA, axis=1)
        dA_cumsum_l = tf.exp(dA_cumsum_l)
        dA_cumsum_l *= tf.expand_dims(last_state, axis=1)
        x = x + dA_cumsum_l

    last_state = x[:, -1]
    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D, last_state


class MambaBlock(tf.keras.layers.Layer):
    def __init__(self, layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states, batch_size, mini_batch_size, stateful, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.layer_id = layer_id
        self.model_internal_dim = model_internal_dim
        self.model_input_dims = model_input_dims
        self.conv_use_bias = conv_use_bias
        self.dense_use_bias = dense_use_bias
        self.conv_kernel_size = conv_kernel_size
        self.delta_t_rank = delta_t_rank
        self.model_states = model_states
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.stateful = stateful

        self.in_projection = tf.keras.layers.Dense(
            self.model_internal_dim * 2,
            input_shape=(self.model_input_dims,), use_bias=False, trainable=True)

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.model_internal_dim,
            use_bias=self.conv_use_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.model_internal_dim,
            data_format='channels_first',
            padding='causal', trainable=True
        )

        self.x_projection = tf.keras.layers.Dense(self.delta_t_rank + self.model_states * 2, use_bias=False, trainable=True)

        self.delta_t_projection = tf.keras.layers.Dense(self.model_internal_dim,
                                               input_shape=(self.delta_t_rank,), use_bias=True, trainable=True)

        self.A = repeat(
            tf.range(1, self.model_states + 1, dtype=tf.float32),
            'n -> d n', d=self.model_internal_dim)

        self.A_log = tf.Variable(
            tf.math.log(self.A),
            trainable=True, dtype=tf.float32,
            name=f"SSM_A_log_{self.layer_id}")

        self.D = tf.Variable(
            np.ones(self.model_internal_dim),
            trainable=True, dtype=tf.float32,
            name=f"SSM_D_{self.layer_id}")

        self.out_projection = tf.keras.layers.Dense(
            self.model_input_dims,
            input_shape=(self.model_internal_dim,),
            use_bias=self.dense_use_bias, trainable=True)
        self.reset_states()

    def reset_states(self):
        self.state = tf.Variable(tf.zeros((9, self.model_internal_dim, self.model_states), dtype=tf.float32), name='state', trainable=False)

    def call(self, x):

        x_and_res = self.in_projection(x)  # shape = (batch, seq_len, 2 * model_internal_dimension)
        (x, res) = tf.split(x_and_res,
                            [self.model_internal_dim,
                             self.model_internal_dim], axis=-1)


        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :self.mini_batch_size]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = tf.nn.swish(x)

        last_state = self.state[:self.batch_size]
        res_state = self.state[self.batch_size:]

        y, y_state = self.ssm(x, last_state=last_state, stateful=self.stateful, L=self.mini_batch_size)

        if self.stateful:
            self.state.assign(tf.concat([y_state, res_state], axis=0))
        y = y * tf.nn.swish(res)
        return self.out_projection(y)

    def ssm(self, x, last_state, stateful):

        (d_in, n) = self.A_log.shape

        A = -tf.exp(tf.cast(self.A_log, tf.float32))  # shape -> (d_in, n)
        D = tf.cast(self.D, tf.float32)
        x_dbl = self.x_projection(x)  # shape -> (batch, seq_len, delta_t_rank + 2*n)
        (delta, B, C) = tf.split(
            x_dbl,
            num_or_size_splits=[self.delta_t_rank, n, n],
            axis=-1)  # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)


        delta = tf.nn.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_input_dim)

        y, y_state = selective_scan(x, delta, A, B, C, D, last_state, stateful, L)

        return y, y_state


class MambaLay(tf.keras.layers.Layer):
    def __init__(self, model_states, projection_expand_factor=2, model_input_dims=2, conv_kernel_size=4, batch_size=9, mini_batch_size=2400, stateful=False, type=tf.float32):
        super(MambaLay, self).__init__()
        layer_id = np.round(np.random.randint(0, 1000), 4)
        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16
        self.model_states = model_states
        conv_use_bias, dense_use_bias = True, True
        self.block = MambaBlock(layer_id, model_input_dims, self.model_internal_dim, conv_use_bias, dense_use_bias,
                            conv_kernel_size, self.delta_t_rank, model_states, batch_size, mini_batch_size, stateful)

    def reset_states(self):
        self.block.reset_states()

    def call(self, x):
        x = self.block(x)
        return x

