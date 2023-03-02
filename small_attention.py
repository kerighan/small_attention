from tensorflow.keras import layers, activations
import tensorflow as tf
import numpy as np


class PositionalEmbedding(layers.Layer):
    def __init__(
        self, n_features, output_dim, mask_zero=False,
        positional_activation=None, **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.mask_zero = mask_zero
        self.n_features = n_features
        self.output_dim = output_dim
        self.supports_masking = True
        self.positional_activation = activations.get(positional_activation)

    def build(self, input_shape):
        seq_len = input_shape[1]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=self.output_dim,)
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=seq_len,
            output_dim=self.output_dim,)

    def call(self, inputs):
        mask = None
        if self.mask_zero:
            mask = tf.expand_dims(tf.math.not_equal(inputs, 0), axis=-1)

        input_length = tf.shape(inputs)[1]

        position_indices = tf.range(input_length, dtype=tf.int32)[None, :]
        position_embeddings = self.positional_activation(
            self.positional_embedding(position_indices))

        x = position_embeddings + self.embedding(inputs)
        if self.mask_zero and mask is not None:
            x *= tf.cast(mask, tf.float32)
        return x

    def compute_mask(self, inputs, mask=None):
        if self.mask_zero:
            return tf.not_equal(inputs, 0)
        else:
            return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "output_dim": self.output_dim,
            "mask_zero": self.mask_zero,
            "positional_activation": activations.serialize(
                self.positional_activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"PositionalEmbedding": PositionalEmbedding}


class PositionalEncoding(layers.Layer):
    def __init__(self, positional_activation=None, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.positional_activation = activations.get(positional_activation)

    def build(self, input_shape):
        seq_len, dim = input_shape[1], input_shape[2]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=seq_len, output_dim=dim)

    def call(self, inputs, mask=None):
        input_length = tf.shape(inputs)[1]
        position_indices = tf.range(
            input_length, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.embedding(position_indices)
        if mask is not None:
            print(mask)
        return position_embeddings + inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "positional_activation": activations.serialize(
                self.positional_activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"PositionalEncoding": PositionalEncoding}



def accuracy(y_true, y_pred):
    # Flatten the prediction and true labels
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
    y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])

    # Create a mask of valid observations
    mask = tf.not_equal(y_true, 0)

    # Compute the accuracy
    acc = tf.cast(tf.equal(y_true[mask], y_pred[mask]), tf.float32)
    acc = tf.reduce_mean(acc)
    return acc


def get_operation(operation):
    if operation == "real":
        return tf.math.real
    elif operation == "imag":
        return tf.math.imag
    elif operation == "abs":
        return tf.math.abs
    elif callable(operation):
        return operation

    raise ValueError("Unknown operation: {}".format(operation))


class SmallAttention(layers.Layer):
    def __init__(
        self, num_heads, intermediate_dim=None, activation="relu", 
        name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = activations.get(activation)
        self._built = False
        self.supports_masking = True

    def _build(self, input_shape):
        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.intermediate_dim is None:
            self.intermediate_dim = in_dim // self.num_heads
        self.final_dim = self.num_heads * self.intermediate_dim

        # attention mechanism
        self.attention_weights = self.add_weight(
            name="attention_weights", shape=(in_dim, self.num_heads),
            initializer="glorot_normal")
        self.temperature = self.add_weight(
            shape=(1, 1, self.num_heads),
            initializer="ones", name="temperature")
        self.values = self.add_weight(
            shape=(in_dim, self.num_heads, self.intermediate_dim),
            initializer="glorot_normal", name="values")
        self.operator = self.add_weight(
            name="operator", shape=(self.final_dim, in_dim),
            initializer="glorot_normal")

        # feed forward layers
        self.feedforward_1 = self.add_weight(
            name="feedforward_1", shape=(2 * in_dim, in_dim),
            initializer="glorot_uniform")
        self.feedforward_1_bias = self.add_weight(
            name="feedforward_1_bias", shape=(1, 1, in_dim),
            initializer="zeros")
        self.feedforward_2 = self.add_weight(
            name="feedforward_2", shape=(in_dim, in_dim),
            initializer="glorot_uniform")
        self.feedforward_2_bias = self.add_weight(
            name="feedforward_2_bias",
            shape=(1, 1, in_dim), initializer="zeros")

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self._built = True

    def call(self, inputs, mask=None):
        if not self._built:
            self._build(inputs.shape)

        att_weights = tf.matmul(inputs, self.attention_weights)
        att_weights -= tf.math.reduce_max(att_weights, axis=-2, keepdims=True)
        att_weights *= self.temperature
        att_weights = tf.exp(att_weights)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            att_weights *= mask
        att_weights /= (
            tf.reduce_sum(att_weights, axis=-2, keepdims=True) + 1e-30)
        att_weights = tf.expand_dims(att_weights, -1)

        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)
        values *= att_weights
        values = tf.reduce_sum(values, axis=1, keepdims=False)
        values = tf.reshape(values, [-1, self.final_dim])

        operator = tf.matmul(values, self.operator)
        operator = operator[:, None, :]
        operator = tf.repeat(operator, repeats=tf.shape(inputs)[1], axis=1)
        concat = tf.concat([inputs, operator], axis=-1)

        # feedforward 1
        res = tf.matmul(concat, self.feedforward_1)
        res = self.activation(res + self.feedforward_1_bias)

        # feedforward 2
        res = tf.matmul(res, self.feedforward_2)
        res = res + self.feedforward_2_bias

        res += inputs  # residual connection
        res = self.layer_norm(res)  # layer normalization

        if mask is not None:
            res *= mask
        return res

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
            "activation": activations.serialize(self.activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"SmallAttention": SmallAttention}


class SmallDecoder(layers.Layer):
    def __init__(
        self, num_heads, intermediate_dim=None, activation="relu",
        name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = activations.get(activation)
        self._built = False
        self.supports_masking = True

    def _build(self, input_shape, vector_shape):
        vector_dim = vector_shape[-1]

        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.intermediate_dim is None:
            self.intermediate_dim = in_dim // self.num_heads
        self.final_dim = self.num_heads * self.intermediate_dim

        # attention mechanism
        self.attention_weights = self.add_weight(
            name="attention_weights", shape=(in_dim, self.num_heads),
            initializer="glorot_normal")
        self.temperature = self.add_weight(
            shape=(1, 1, self.num_heads),
            initializer="ones", name="temperature")
        self.values = self.add_weight(
            shape=(in_dim, self.num_heads, self.intermediate_dim),
            initializer="glorot_normal", name="values")
        self.operator = self.add_weight(
            name="operator", shape=(self.final_dim, vector_dim),
            initializer="glorot_normal")

        # feed forward layers
        self.feedforward_1 = self.add_weight(
            name="feedforward_1", shape=(vector_dim + in_dim, in_dim),
            initializer="glorot_uniform")
        self.feedforward_1_bias = self.add_weight(
            name="feedforward_1_bias", shape=(1, 1, in_dim),
            initializer="zeros")
        self.feedforward_2 = self.add_weight(
            name="feedforward_2", shape=(in_dim, in_dim),
            initializer="glorot_uniform")
        self.feedforward_2_bias = self.add_weight(
            name="feedforward_2_bias",
            shape=(1, 1, in_dim), initializer="zeros")

        self.causal_mask = np.tril(
            np.ones((self.seq_len, self.seq_len))
        )[None, :, :, None].astype(np.float32)

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self._built = True

    def call(self, inputs, vector, mask=None):
        if not self._built:
            self._build(inputs.shape, vector.shape)

        # compute attention weights
        att_weights = tf.matmul(inputs, self.attention_weights)
        att_weights -= tf.math.reduce_max(att_weights, axis=-2, keepdims=True)
        att_weights *= self.temperature
        att_weights = tf.exp(att_weights)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            att_weights *= mask
        att_weights = tf.repeat(
            att_weights[:, :, None, :], repeats=self.seq_len, axis=2)
        att_weights *= self.causal_mask
        att_weights /= (
            tf.reduce_sum(att_weights, axis=-2, keepdims=True) + 1e-30)
        att_weights = tf.expand_dims(att_weights, -1)

        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)[:, None]
        values *= att_weights
        values = tf.reduce_sum(values, axis=2, keepdims=False)
        values = tf.reshape(values, [-1, self.seq_len, self.final_dim])
        operator = tf.matmul(values, self.operator)

        if len(vector.shape) == 3:
            vector = tf.reduce_mean(vector, axis=1, keepdims=True)
        else:
            vector = vector[:, None, :]
        operator += vector
        concat = tf.concat([inputs, operator], axis=-1)

        # feedforward 1
        res = tf.matmul(concat, self.feedforward_1)
        res = self.activation(res + self.feedforward_1_bias)

        # feedforward 2
        res = tf.matmul(res, self.feedforward_2)
        res = res + self.feedforward_2_bias

        res += inputs  # residual connection
        res = self.layer_norm(res)  # layer normalization
        return res

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
            "activation": activations.serialize(self.activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"SmallDecoder": SmallDecoder}


class SmallPooling(layers.Layer):
    def __init__(
        self, num_heads, out_dim=None,
        activation=None, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.activation = activations.get(activation)

    def build(self, input_shape):
        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.out_dim is None:
            self.out_dim = in_dim

        intermediate_dim = self.out_dim // self.num_heads
        self.flatten_dim = self.num_heads * intermediate_dim

        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(in_dim, self.num_heads))
        self.values = self.add_weight(
            name="values",
            shape=(in_dim, self.num_heads, intermediate_dim))
        self.temperature = self.add_weight(
            name="temperature",
            shape=(1, 1, self.num_heads), initializer="ones")
        self.feedforward = self.add_weight(
            name="feedforward", shape=(self.flatten_dim, self.out_dim),
            initializer="glorot_normal")
        self.feedforward_bias = self.add_weight(
            name="feedforward_bias", shape=(1, self.out_dim),
            initializer="zeros")

    def call(self, inputs, mask=None):
        att_weights = tf.matmul(inputs, self.attention_weights)
        att_weights *= self.temperature
        att_weights = tf.exp(att_weights)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            att_weights *= mask

        att_weights /= (
            tf.reduce_sum(att_weights, axis=-2, keepdims=True) + 1e-30)
        att_weights = tf.expand_dims(att_weights, -1)

        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)
        values *= att_weights
        values = tf.reduce_sum(values, axis=1)
        values = tf.reshape(values, [-1, self.flatten_dim])

        # feedforward
        res = tf.matmul(values, self.feedforward)
        res += self.feedforward_bias
        res = self.activation(res)
        return res

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "out_dim": self.out_dim,
            "activation": activations.serialize(self.activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"SmallPooling": SmallPooling}
