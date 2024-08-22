import tensorflow as tf
from Mamba import FiLM, TemporalFiLM, ResidualBlock, S6
from S4D import S4D
import numpy as np
import math
    
def create_model_ED_original(input_dim, units, b_size=600, comp=''):

    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dim), name='input')
    encoder_inputs, decoder_inputs = tf.split(inp, [input_dim//2, input_dim//2], axis=-1)

    encoder_inputs = tf.expand_dims(encoder_inputs, axis=-1)
    state_h = tf.keras.layers.Conv1D(units//3, input_dim//2, name='Conv_h')(encoder_inputs)
    state_c = tf.keras.layers.Conv1D(units//3, input_dim//2, name='Conv_c')(encoder_inputs)

    if comp == 'CL1B':
    
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs')

        z_h = tf.keras.layers.Dense(units//3, name='Dense_cond_h')(z)
        z_c = tf.keras.layers.Dense(units//3, name='Dense_cond_c')(z)

        states_h = tf.keras.layers.Add()([state_h[:, 0, :], z_h])
        states_c = tf.keras.layers.Add()([state_c[:, 0, :], z_c])
        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs2')
        z2_ = tf.expand_dims(z2, axis=1)

        z2_h = tf.keras.layers.GRU(units//3, return_sequences=False, return_state=False, stateful=True, name='GRU_cond_h')(z2_)
        z2_c = tf.keras.layers.GRU(units//3, return_sequences=False, return_state=False, stateful=True, name='GRU_cond_c')(z2_)
        states_h = tf.keras.layers.Add()([states_h, z2_h])
        states_c = tf.keras.layers.Add()([states_c, z2_c])
     
    elif comp == 'LA2A':
        # peak
        z = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs')

        z_h = tf.keras.layers.Dense(units//3, name='Dense_cond_h')(z)
        z_c = tf.keras.layers.Dense(units//3, name='Dense_cond_c')(z)

        states_h = tf.keras.layers.Add()([state_h[:, 0, :], z_h])
        states_c = tf.keras.layers.Add()([state_c[:, 0, :], z_c])
        # switch
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs2')
        z2_ = tf.expand_dims(z2, axis=1)

        z2_h = tf.keras.layers.GRU(units//3, return_sequences=False, return_state=False, stateful=True, name='GRU_cond_h')(z2_)
        z2_c = tf.keras.layers.GRU(units//3, return_sequences=False, return_state=False, stateful=True, name='GRU_cond_c')(z2_)
        states_h = tf.keras.layers.Add()([states_h, z2_h])
        states_c = tf.keras.layers.Add()([states_c, z2_c])

    states_h = tf.keras.layers.Dense(units-1, name='encoder_states_dense_h')(states_h)
    states_c = tf.keras.layers.Dense(units-1, name='encoder_states_dense_c')(states_c)
    encoder_states = [states_h, states_c]

    decoder_inputs = tf.expand_dims(decoder_inputs, axis=1)
    x = tf.keras.layers.LSTM(units-1, return_sequences=False, return_state=False, name='LSTM_decoder')(decoder_inputs, initial_state=encoder_states)
    x = tf.keras.layers.Dense(2, activation='sigmoid', name='DenseLay')(x)
    x = tf.keras.layers.Dense(1)(x)

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM-CNN-ED')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM-CNN-ED')

    model.summary()

    return model

def create_model_LSTM_baseline(input_dim, b_size=600, comp=''):
    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dim), name='input')

    x = tf.keras.layers.Dense(1)(inp)


    if comp == 'CL1B':
        z = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs')
        c = tf.concat([x, z], axis=-1)
        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs2')
        x = tf.concat([z2, c], axis=-1)
    elif comp == 'LA2A':
        z = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs')
        c = tf.concat([x, z], axis=-1)
        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs2')
        x = tf.concat([z2, c], axis=-1)

    x = tf.expand_dims(x, axis=1)
    x = tf.keras.layers.LSTM(14, return_sequences=False, return_state=False, stateful=True, name='LSTM')(x)
    x = tf.keras.layers.Dense(2)(x)

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    #x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM-baseline')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, inp], outputs=x, name='LSTM-baseline')

    model.summary()

    return model

def create_model_LSTM(input_dim, b_size=600, comp=''):
    
    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dim), name='input')

    x = tf.keras.layers.Dense(2)(inp)
    x = tf.expand_dims(x, axis=1)
    x = tf.keras.layers.LSTM(6, return_sequences=False, return_state=False, stateful=True, name='LSTM')(x)
    x = tf.keras.layers.Dense(2)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=1)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    
    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs')
        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    ###########
    x = tf.expand_dims(x, axis=1)

    x = tf.keras.layers.LSTM(6, return_sequences=False, return_state=False, stateful=True, name='LSTM2')(x)
  
    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])


    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM9')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM9')

    model.summary()

    return model


def create_model_ED_CNN(input_dim, units, b_size=600, comp=''):##mettere z su states?
    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dim), name='input')
    #encoder_inputs, decoder_inputs = tf.split(inp, [input_dim - 1, 1], axis=-1)

    x = tf.expand_dims(inp, axis=-1)
    h = tf.keras.layers.Conv1D(units//2, input_dim, name='Conv_h')(x)
    c = tf.keras.layers.Conv1D(units//2, input_dim, name='Conv_c')(x)
    h = tf.squeeze(h, axis=1)
    c = tf.squeeze(c, axis=1)
    x = tf.squeeze(x, axis=-1)

    x = tf.keras.layers.Dense(2)(x)
    x = tf.expand_dims(x, axis=1)
    x = tf.keras.layers.LSTM(units//2, return_sequences=False, return_state=False, name='LSTM_decoder')(x, initial_state=[h, c])

    x = tf.keras.layers.Dense(2)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=1)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    
    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs')
        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    ###########

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM-ED')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='LSTM-ED')

    model.summary()

    return model


def create_model_S4D(input_dim, units, b_size=600, comp=''):
    
    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dim), name='input')
    x = tf.keras.layers.Dense(2)(inp)#####
    x = S4D(units)(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    f = tf.keras.layers.Input(batch_shape=(b_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=1)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs')
        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    
    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs')
        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    ###########

    x = S4D(units)(x)
    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = tf.keras.layers.Multiply()([inp[:,-1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S4D')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S4D')

    model.summary()

    return model



def create_model_Mamba(b_size, input_dims=64, model_input_dims=4, conv_use_bias=True, dense_use_bias=True,
               projection_expand_factor=2, conv_kernel_size=4, model_states=8, comp=''):

    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dims), name='input_ids')
    x = tf.keras.layers.Dense(2)(inp)
    x = tf.expand_dims(x, axis=1)

    model_internal_dim = int(projection_expand_factor * model_input_dims)
    delta_t_rank = math.ceil(model_input_dims / 2)  # 16

    layer_id = np.round(np.random.randint(0, 1000), 4)

    x = ResidualBlock(layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size,
                      delta_t_rank, model_states, name=f"Residual_{0}")(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)
    x = tf.squeeze(x, axis=1)

    ###########
    f = tf.keras.layers.Input(batch_shape=(b_size, 128, 1), name='features_input')
    features = tf.keras.layers.Conv1D(2, 128)(f)
    features = tf.squeeze(features, axis=1)

    if comp == 'CL1B':
        # threshold and ratio
        z = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs')

        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # attack and release
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 2), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    
    elif comp == 'LA2A':
        # peak reduction
        z = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs')
        c = tf.concat([z, features], axis=-1)
        x = FiLM(2)(x, c)

        # switch
        z2 = tf.keras.layers.Input(batch_shape=(b_size, 1), name='params_inputs2')
        c = tf.concat([z2, features], axis=-1)
        c = tf.expand_dims(c, axis=1)
        x = TemporalFiLM(2)(x, c)
    ###########

    model_input_dims = 2
    x = tf.expand_dims(x, axis=1)
    layer_id = np.round(np.random.randint(0, 1000), 4)
    x = ResidualBlock(layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size,
                      delta_t_rank, model_states, name=f"Residual_{1}")(x)

    x = tf.keras.layers.Dense(2, activation=tf.nn.gelu)(x)####

    x = tf.squeeze(x, axis=1)

    x = tf.keras.layers.Dense(1)(x)
    x = tf.keras.layers.Multiply()([inp[:, -1], x])

    if comp == 'CL1B':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S6')
    elif comp == 'LA2A':
        model = tf.keras.models.Model(inputs=[z, z2, f, inp], outputs=x, name='S6')
    model.summary()

    return model
