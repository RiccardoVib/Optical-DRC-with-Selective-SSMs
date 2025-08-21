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
from Mamba import MambaLay
from S4D import S4D
from Layers import FiLM, TemporalFiLM


def create_model_ED_baseline(mini_batch_size, input_dim, units, b_size=600, comp='', stateful=True):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')
    encoder_inputs, decoder_inputs = tf.split(inp, [input_dim // 2, input_dim // 2], axis=-1)

    state_h = tf.keras.layers.Conv1D(units // 3, input_dim // 2, name='Conv_h')(encoder_inputs)
    state_c = tf.keras.layers.Conv1D(units // 3, input_dim // 2, name='Conv_c')(encoder_inputs)

    if comp == 'CL1B':

        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')

        z_h = tf.keras.layers.Dense(units // 3, name='Dense_cond_h')(z)
        z_c = tf.keras.layers.Dense(units // 3, name='Dense_cond_c')(z)

        states_h = tf.keras.layers.Add()([state_h[:, 0, :], z_h])
        states_c = tf.keras.layers.Add()([state_c[:, 0, :], z_c])
        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        z2_ = tf.expand_dims(z2, axis=1)

        z2_h = tf.keras.layers.GRU(units // 3, return_sequences=True, stateful=stateful,
                                   name='GRU_cond_h')(z2_)
        z2_c = tf.keras.layers.GRU(units // 3, return_sequences=True, stateful=stateful,
                                   name='GRU_cond_c')(z2_)
        states_h = tf.keras.layers.Add()([states_h, z2_h])
        states_c = tf.keras.layers.Add()([states_c, z2_c])

    elif comp == 'LA2A':
        # peak
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')

        z_h = tf.keras.layers.Dense(units // 3, name='Dense_cond_h')(z)
        z_c = tf.keras.layers.Dense(units // 3, name='Dense_cond_c')(z)

        states_h = tf.keras.layers.Add()([state_h[:, 0, :], z_h])
        states_c = tf.keras.layers.Add()([state_c[:, 0, :], z_c])
        # switch
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        z2_ = tf.expand_dims(z2, axis=1)

        z2_h = tf.keras.layers.GRU(units // 3, return_sequences=True, stateful=stateful,
                                   name='GRU_cond_h')(z2_)
        z2_c = tf.keras.layers.GRU(units // 3, return_sequences=True, stateful=stateful,
                                   name='GRU_cond_c')(z2_)
        states_h = tf.keras.layers.Add()([states_h, z2_h])
        states_c = tf.keras.layers.Add()([states_c, z2_c])

    states_h = tf.keras.layers.Dense(units - 1, name='encoder_states_dense_h')(states_h)
    states_c = tf.keras.layers.Dense(units - 1, name='encoder_states_dense_c')(states_c)
    encoder_states = [states_h, states_c]

    decoder_inputs = tf.expand_dims(decoder_inputs, axis=1)
    x = tf.keras.layers.LSTM(units - 1, return_sequences=True, name='LSTM_decoder')(decoder_inputs, initial_state=encoder_states)
    x = tf.keras.layers.Dense(2, activation='sigmoid', name='DenseLay')(x)
    x = tf.keras.layers.Dense(1)(x)

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM-CNN-ED')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM-CNN-ED')

    model.summary()

    return model


def create_model_LSTM_baseline(mini_batch_size, input_dim, b_size=600, comp='', stateful=True):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    x = tf.keras.layers.Dense(1)(inp)

    if comp == 'CL1B':
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')
        c = tf.concat([x, z], axis=-1)
        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        c = tf.concat([z2, c], axis=-1)
    elif comp == 'LA2A':
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')
        c = tf.concat([x, z], axis=-1)
        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        c = tf.concat([z2, c], axis=-1)

    x = tf.expand_dims(c, axis=1)
    x = tf.keras.layers.LSTM(14, return_sequences=True, stateful=stateful, name='LSTM')(x)
    x = tf.keras.layers.Dense(2)(x)

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM16-baseline')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM16-baseline')

    model.summary()

    return model


def create_model_LSTM(mini_batch_size, input_dim, b_size=600, comp='', stateful=True):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    x = tf.keras.layers.Dense(2)(inp)
    x = tf.keras.layers.LSTM(6, return_sequences=True, stateful=stateful, name='LSTM')(x)
    x = tf.keras.layers.Dense(2)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=2)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        c2 = tf.concat([z2, features], axis=-1)
        x = TemporalFiLM(2)(x, c2)

    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        c = tf.concat([z, z2], axis=-1)
        c = tf.concat([c, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        c2 = features
        x = TemporalFiLM(2)(x, c2)
    ###########

    x = tf.keras.layers.LSTM(6, return_sequences=True, stateful=stateful, name='LSTM2')(x)

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM9')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM9')

    model.summary()

    return model


def create_model_ED_CNN(mini_batch_size, input_dim, units, b_size=600, comp='', stateful=True):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')

    x = tf.expand_dims(inp[:,0,:], axis=-1)
    h = tf.keras.layers.Conv1D(units // 2, input_dim, name='Conv_h')(x)
    c = tf.keras.layers.Conv1D(units // 2, input_dim, name='Conv_c')(x)
    h = tf.squeeze(h, axis=1)
    c = tf.squeeze(c, axis=1)
    x = tf.squeeze(x, axis=-1)

    x = tf.keras.layers.Dense(2)(x)
    x = tf.expand_dims(x, axis=1)
    x = tf.keras.layers.LSTM(units // 2, return_sequences=True, name='LSTM_decoder')(x, initial_state=[h, c])

    x = tf.keras.layers.Dense(2)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=2)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        c2 = tf.concat([z2, features], axis=-1)
        x = TemporalFiLM(2, stateful=stateful)(x, c2)

    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        c = tf.concat([z, z2], axis=-1)
        c = tf.concat([c, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        c2 = features
        x = TemporalFiLM(2)(x, c2)
    ###########

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM-ED')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM-ED')

    model.summary()

    return model

def create_model_S4D(mini_batch_size, input_dim, units, b_size=600, comp='', stateful=False):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')
    x = tf.keras.layers.Dense(2)(inp)

    x = S4D(model_states=units, model_input_dims=2, batch_size=b_size, mini_batch_size=mini_batch_size, stateful=stateful, hippo=True)(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=2)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        c2 = tf.concat([z2, features], axis=-1)
        x = TemporalFiLM(2, stateful=stateful)(x, c2)

    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        c = tf.concat([z, z2], axis=-1)
        c = tf.concat([c, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        c2 = features
        x = TemporalFiLM(2, stateful=stateful)(x, c2)
    ###########

    x = S4D(model_states=units, model_input_dims=2, batch_size=b_size, mini_batch_size=mini_batch_size, stateful=stateful, hippo=True)(x)
    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S4D')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S4D')

    model.summary()

    return model

def create_model_S6(mini_batch_size, input_dim, units, b_size=600, comp='', stateful=False):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dim), name='input')
    x = tf.keras.layers.Dense(2)(inp)

    x = S6(model_states=units, model_input_dims=2, batch_size=b_size, mini_batch_size=mini_batch_size, stateful=stateful)(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=2)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        c2 = tf.concat([z2, features], axis=-1)
        x = TemporalFiLM(2, stateful=stateful)(x, c2)

    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        c = tf.concat([z, z2], axis=-1)
        c = tf.concat([c, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        c2 = features
        x = TemporalFiLM(2, stateful=stateful)(x, c2)
    ###########

    x = S6(model_states=units, model_input_dims=2, batch_size=b_size, mini_batch_size=mini_batch_size, stateful=stateful)(x)
    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S4D')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S4D')

    model.summary()

    return model


def create_model_Mamba(mini_batch_size, b_size, input_dims=64, model_input_dims=1,
                       projection_expand_factor=2, conv_kernel_size=4, model_states=8, comp='', stateful=False):
    inp = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, input_dims), name='input_ids')
    x = tf.keras.layers.Dense(2)(inp)  #####

    x = MambaLay(model_states=model_states, projection_expand_factor=projection_expand_factor, model_input_dims=2,
                 conv_kernel_size=conv_kernel_size, batch_size=b_size, mini_batch_size=mini_batch_size, stateful=stateful)(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    ###########
    f = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=2)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 2), name='params_inputs2')
        c2 = tf.concat([z2, features], axis=-1)
        x = TemporalFiLM(in_size=2, stateful=stateful)(x, c2)

    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs')
        z2 = tf.keras.layers.Input(batch_shape=(b_size, mini_batch_size, 1), name='params_inputs2')
        c = tf.concat([z, z2], axis=-1)
        c = tf.concat([c, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        c2 = features
        x = TemporalFiLM(in_size=2, stateful=stateful)(x, c2)
    ###########

    x = MambaLay(model_states=model_states, projection_expand_factor=projection_expand_factor,
                 model_input_dims=model_input_dims,
                 conv_kernel_size=conv_kernel_size, batch_size=b_size, mini_batch_size=mini_batch_size, stateful=stateful)(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)  ####

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)

    x = tf.keras.layers.Multiply()([inp[:, :, -1:], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='Mamba')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='Mamba')
    model.summary()

    return model