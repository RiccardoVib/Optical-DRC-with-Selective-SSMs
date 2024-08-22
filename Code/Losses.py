import tensorflow as tf
from keras import backend as K



class DCMSE(tf.keras.losses.Loss):
    def __init__(self, name="DCMSE", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        loss = K.mean(K.square(K.mean(y_true, axis=0) - K.mean(y_pred, axis=0)))
        energy = K.mean(K.square(y_true)) + 0.00001
        loss = tf.divide(loss, energy)

        DC = tf.divide(K.mean(K.square(K.add(y_pred, -y_true))), K.mean(K.square(y_true) + 0.00001))
        MSE = tf.keras.losses.MeanSquaredError(y_pred, y_true)
        return K.add(DC, MSE)

    def get_config(self):
        config = {
        }
        base_config = super().get_config()
        return {**base_config, **config}


class DC(tf.keras.losses.Loss):
    def __init__(self, name="DC", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        loss = K.mean(K.square(K.mean(y_true, axis=0) - K.mean(y_pred, axis=0)))
        energy = K.mean(K.square(y_true)) + 0.00001
        loss = tf.divide(loss, energy)

        return tf.divide(K.mean(K.square(y_pred - y_true)), K.mean(K.square(y_true) + 0.00001))

    def get_config(self):
        config = {
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ESR(tf.keras.losses.Loss):
    def __init__(self, name="ESR", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # return K.log(tf.divide(K.square(tf.norm((y_pred - y_true), ord=2)), K.square(tf.norm((y_pred), ord=2)) + 0.0000001) +1)
        return tf.divide(K.mean(K.square(y_pred - y_true)), K.mean(K.square(y_true) + 0.00001))

    def get_config(self):
        config = {
        }
        base_config = super().get_config()
        return {**base_config, **config}


class LogMSE(tf.keras.losses.Loss):
    def __init__(self, name="LogMSE", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return K.log(tf.keras.losses.mean_squared_error(y_true, y_pred) + 1)

    def get_config(self):
        config = {

        }
        base_config = super().get_config()
        return {**base_config, **config}


class Normalized_AE(tf.keras.losses.Loss):
    def __init__(self, name="Normalized_AE", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.divide(K.square(y_pred - y_true), K.square(y_true) + 0.00000001))  # K.sum

    def get_config(self):
        config = {
        }
        base_config = super().get_config()
        return {**base_config, **config}


class STFT(tf.keras.losses.Loss):
    def __init__(self, batch_size, m=[32, 64, 128, 256, 512, 1024], name="CustomLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])

        loss = 0
        log_loss = 0
        for i in range(len(self.m)):
            pad_amount = int(self.m[i] // 2)  # Symmetric even padding like librosa.
            pads = [[0, 0], [pad_amount, pad_amount]]
            #pads = [pad_amount, pad_amount]
            y_true = tf.pad(y_true, pads, mode='constant', constant_values=0)
            y_pred = tf.pad(y_pred, pads, mode='constant', constant_values=0)

            Y_true = K.abs(
                tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))
            Y_pred = K.abs(
                tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4,
                               pad_end=False))

            Y_true = tf.pow(K.abs(Y_true), 2)
            Y_pred = tf.pow(K.abs(Y_pred), 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.norm((l_true - l_pred), ord=1) / Y_true.shape[0]

            log_loss += tf.norm((Y_true - Y_pred), ord=1) / Y_true.shape[0]

        return 0.0001*((loss + log_loss) / len(self.m)) + mse

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}  


class STFT_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512, 1024], name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):

        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        loss = 0
        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            Y_true = K.pow(K.abs(Y_true), 2)
            Y_pred = K.pow(K.abs(Y_pred), 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.norm((l_true - l_pred), ord=1) + tf.norm((Y_true - Y_pred), ord=1)

        return (loss/len(self.m)) + mse

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}