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
from einops import rearrange, repeat
import numpy as np
import math

'''
code adapted from https://github.com/state-spaces/mamba & https://github.com/PeaBrane/mamba-tiny
'''

_EPS = tf.constant(1e-12, dtype=tf.float32)


def selective_scan(u, delta, A, B, C, D, last_state, stateful):
    u = tf.convert_to_tensor(u)
    delta = tf.cast(delta, u.dtype)
    A = tf.cast(A, u.dtype)
    B = tf.cast(B, u.dtype)
    C = tf.cast(C, u.dtype)
    D = tf.cast(D, u.dtype)

    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    padded_dA = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])
    cumulative_A = tf.math.cumsum(padded_dA, axis=1)
    cumulative_A_exp = tf.exp(cumulative_A)

    states = tf.cumsum(dB_u / (cumulative_A + _EPS), axis=1) * cumulative_A

    if stateful and last_state is not None:
        initial_A = tf.exp(tf.cumsum(dA, axis=1))
        states = states + initial_A * last_state[:, None, :, :]

    final_state = states[:, -1]
    output = tf.einsum("bldn,bln->bld", states, C)
    return output + u * D, final_state


class MambaLay(tf.keras.layers.Layer):
    def __init__(self,  
        model_input_dims,
        projection_expand_factor,
        conv_kernel_size,
        delta_t_rank,
        model_states,
        batch_size,
        mini_batch_size,
        conv_use_bias=True,
        dense_use_bias=True,
        stateful=True
        ):
        super(MambaBlock, self).__init__(**kwargs)
        self.model_input_dims = int(model_input_dims)
        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16            
        self.conv_kernel_size = int(conv_kernel_size)
        self.model_states = int(model_states)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.stateful = bool(stateful)
        self._state: Optional[tf.Variable] = None

        self.in_projection = tf.keras.layers.Dense(
            2 * self.model_internal_dim,
            use_bias=False,
        )
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.model_internal_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.model_internal_dim,
            use_bias=conv_use_bias,
            data_format="channels_first",
            padding="causal",
        )
        self.x_projection = tf.keras.layers.Dense(
            self.delta_t_rank + 2 * self.model_states,
            use_bias=False,
        )
        self.delta_t_projection = tf.keras.layers.Dense(
            self.model_internal_dim,
            use_bias=True,
        )
        self.out_projection = tf.keras.layers.Dense(
            self.model_input_dims,
            use_bias=dense_use_bias,
        )
            
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
            use_bias=self.dense_use_bias)
        self.reset_states()

    def _ensure_state(self, batch_size):
        shape = (batch_size, self.model_internal_dim, self.model_states)
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

    def call(self, x):

        x_and_res = self.in_projection(x)  # shape = (batch, seq_len, 2 * model_internal_dimension)
        (x, res) = tf.split(x_and_res,
                            [self.model_internal_dim,
                             self.model_internal_dim], axis=-1)


        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :self.mini_batch_size]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = tf.nn.swish(x)

        state = self._ensure_state(tf.shape(x)[0])
        y, final_state = self.ssm(
            x,
            last_state=state,
            stateful=self.stateful,
        )
        if self.stateful:
            state.assign(final_state)

        y = y * tf.nn.swish(residual)
        return self.out_projection(y)

    def ssm(self, x, last_state, stateful):

        A = -tf.exp(tf.cast(self.A_log, x.dtype))  # shape -> (d_in, n)
        x_dbl = self.x_projection(x)  # shape -> (batch, seq_len, delta_t_rank + 2*n)

        delta, B, C = tf.split(
            x_dbl,
            [self.delta_t_rank, self.model_states, self.model_states],
            axis=-1,
        )   # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)
               
        delta = tf.nn.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_input_dim)

       return selective_scan(
            x,
            delta,
            A,
            B,
            C,
            self.D,
            last_state,
            stateful,
        )

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

