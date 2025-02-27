import tensorflow as tf

class FiLM(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias)
        self.glu = GLU(in_size=self.in_size)

    def call(self, x, c):
        c = self.dense(c)
        a, b = tf.split(c, 2, axis=self.dim)
        x = tf.multiply(a, x)
        x = tf.add(x, b)
        x = self.glu(x)
        return x

class TemporalFiLM(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, stateful=False, **kwargs):
        super(TemporalFiLM, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.gru = tf.keras.layers.GRU(self.in_size * 2, stateful=stateful, return_sequences=True, use_bias=bias)
        self.glu = GLU(in_size=self.in_size)

    def reset_states(self):
        self.gru.reset_states()
    def call(self, x, c):
        c = self.gru(c)
        a, b = tf.split(c, 2, axis=self.dim)
        x = tf.multiply(a, x)
        x = tf.add(x, b)
        x = self.glu(x)
        return x