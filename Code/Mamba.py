import tensorflow as tf
from einops import rearrange, repeat
import numpy as np
import math

#### from https://github.com/state-spaces/mamba & https://github.com/PeaBrane/mamba-tiny


def selective_scan(u, delta, A, B, C, D, last_state, stateful, L):
    # first step of A_bar = exp(ΔA), i.e., ΔA
    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    if stateful == False:
        #dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :] ### add zero in the first spot since starting with state=0, and in last
        dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]], constant_values=0) ### add zero in the first spot since starting with state=0, and in last
    else:
        dA_cumsum = tf.concat([last_state[:, np.newaxis, :, :], dA[:, 1:]], axis=1)  ##### add state in the first spot since starting with state=0

    #dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip along axis 1
    dA_cumsum = tf.exp(dA_cumsum)
    #dA_cumsum = tf.math.cumprod(dA_cumsum, axis=1)

    xs = []

    for i in range(L):
        last_state = dA_cumsum[:, i] * last_state + dB_u[:, i]
        xs.append(last_state)
    # Cumulative sum along all the input tokens, parallel prefix sum,
    # calculates dA for all the input tokens parallely
    #dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)

    # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
    #dA_cumsum = tf.exp(dA_cumsum)
    #dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip back along axis 1

    #x = dB_u * dA_cumsum
    # 1e-12 to avoid division by 0
    #x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)

    #if stateful == True:
    #    last_state = x[:, -1:]

    x = tf.stack(xs, axis=1)
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

        # this layer takes in current token 'x'
        # and outputs the input-specific Δ, B, C (according to S6)
        self.x_projection = tf.keras.layers.Dense(self.delta_t_rank + self.model_states * 2, use_bias=False, trainable=True)

        # this layer projects Δ from delta_t_rank to the mamba internal
        # dimension
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
        #x = tf.Variable(tf.random.normal([self.batch_size, 2048, 2]), dtype='float32')
        self.reset_states()

        #if self.stateful == False:
        #    self.state = None
    def reset_states(self):
        self.state = tf.Variable(tf.zeros((9, self.model_internal_dim, self.model_states), dtype=tf.float32), name='state', trainable=False)
        #pass
    def call(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba pape.
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """

        (batch_size, seq_len, dimension) = x.shape

        x_and_res = self.in_projection(x)  # shape = (batch, seq_len, 2 * model_internal_dimension)
        (x, res) = tf.split(x_and_res,
                            [self.model_internal_dim,
                             self.model_internal_dim], axis=-1)


        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = tf.nn.swish(x)

        last_state = self.state[:self.batch_size]
        res_state = self.state[self.batch_size:]

        y, y_state = self.ssm(x, last_state=last_state, stateful=self.stateful, L=self.mini_batch_size)

        if self.stateful:
            self.state.assign(tf.concat([y_state, res_state], axis=0))
        y = y * tf.nn.swish(res)
        return self.out_projection(y)

    def ssm(self, x, last_state, stateful, L):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper
            - run_SSM(A, B, C, u) in The Annotated S4
            Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

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



class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states, batch_size, mini_batch_size, stateful, **kwargs):
        super().__init__(**kwargs)
        self.mixer = MambaBlock(layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states, batch_size, mini_batch_size, stateful)
        #self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)######
    def reset_states(self):
        self.mixer.reset_states()
    def call(self, x):
        """
        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        #return self.mixer(self.norm(x)) + x#####
        return self.mixer(x)




class MambaLay(tf.keras.layers.Layer):
    def __init__(self, model_states, projection_expand_factor=2, model_input_dims=2, conv_kernel_size=4, batch_size=9, mini_batch_size=2400, stateful=False, type=tf.float32):
        super(MambaLay, self).__init__()
        layer_id = np.round(np.random.randint(0, 1000), 4)
        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16
        self.model_states = model_states
        conv_use_bias, dense_use_bias = True, True
        self.block = ResidualBlock(layer_id, model_input_dims, self.model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size,
                      self.delta_t_rank, model_states, batch_size, mini_batch_size, stateful, name=f"Residual_{0}")
        #self.dense = tf.keras.layers.Dense(units, activation=tf.nn.gelu, trainable=trainable)
        #self.dense = tf.keras.layers.Dense(model_states, activation=tf.nn.gelu)
        #self.out = tf.keras.layers.Dense(1, activation=tf.nn.gelu)

    def reset_states(self):
        self.block.reset_states()

    def call(self, x):
        x = self.block(x)
        #x = self.dense(x)
        #x = self.out(x)
        return x

