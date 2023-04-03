from tensorflow.keras import layers, activations, initializers
from tensorflow.keras import backend as K
from keras.layers.pooling.base_global_pooling1d import GlobalPooling1D
import tensorflow as tf
import numpy as np


def clone_initializer(initializer):
    if not isinstance(initializer, initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


class SmallAttention(layers.Layer):
    """
    A small multi-head self-attention layer.
    Arguments:
    - num_heads: The number of attention heads.
    - intermediate_dim: The dimension of the intermediate projection.
                        If `None`, it defaults to `in_dim // num_heads`.
    - activation: The activation function to use in the feedforward layers.
                  Defaults to 'relu'.
    - dropout: The rate of dropout to apply. Defaults to 0.
    - name: A string for the name of the layer (optional).
    Attributes:
    - num_heads: The number of attention heads.
    - intermediate_dim: The dimension of the intermediate projection.
    - activation: The activation function to use in the feedforward layers.
    - dropout: The rate of dropout to apply.
    - _built: A boolean indicating whether the layer has been built.
    - supports_masking: A boolean indicating whether to support masking.
    - seq_len: The length of the input sequence.
    - in_dim: The dimension of the input sequence.
    - final_dim: The final dimension after concatenating the attention heads.
    - attention_weights: The weights to compute attention scores.
    - temperature: The scaling factor for the attention scores.
    - values: The learned projections for the values.
    - operator: The learned projection to combine the attention heads.
    - feedforward_layer_1: The weights for the first feedforward layer.
    - feedforward_layer_1_bias: The bias for the first feedforward layer.
    - feedforward_layer_2: The weights for the second feedforward layer.
    - feedforward_layer_2_bias: The bias for the second feedforward layer.
    - layer_norm: The layer normalization layer.
    """
    def __init__(
        self,
        num_heads,
        intermediate_dim=None,
        activation="relu",
        dropout=0,
        name=None,
        merge_mode="sum",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = activations.get(activation)
        self.dropout = dropout
        self.merge_mode = merge_mode

        self._built = False
        self.supports_masking = True

    def _build(self, input_shape):
        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.intermediate_dim is None:
            self.intermediate_dim = in_dim // self.num_heads
        self.final_dim = self.num_heads * self.intermediate_dim

        # dropouts
        self._attention_dropout = layers.Dropout(rate=self.dropout)
        self._feedforward_dropout = layers.Dropout(rate=self.dropout)

        # attention mechanism
        dtype = tf.float32
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(in_dim, self.num_heads),
            initializer=clone_initializer("orthogonal"),
            dtype=dtype)
        self.temperature = self.add_weight(
            name="temperature", shape=(1, 1, self.num_heads),
            initializer="ones", dtype=dtype)

        initializer = "glorot_uniform"
        self.values = self.add_weight(
            name="values",
            shape=(in_dim, self.num_heads, self.intermediate_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)

        # feed forward layers
        initializer = "glorot_normal"
        ff_in_dim = in_dim if self.merge_mode == "sum" else in_dim + self.final_dim
        self.feedforward_layer_1 = self.add_weight(
            name="feedforward_layer_1",
            shape=(ff_in_dim, in_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)
        self.feedforward_layer_1_bias = self.add_weight(
            name="feedforward_layer_1_bias",
            shape=(1, 1, in_dim),
            initializer="zeros",
            dtype=dtype)
        self.feedforward_layer_2 = self.add_weight(
            name="feedforward_layer_2",
            shape=(in_dim, in_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)
        self.feedforward_layer_2_bias = self.add_weight(
            name="feedforward_layer_2_bias",
            shape=(1, 1, in_dim),
            initializer="zeros",
            dtype=dtype)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, axis=-1)

        self._built = True

    def call(self, inputs, mask=None):
        if not self._built:
            self._build(inputs.shape)

        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)

        # compute multihead attention weights
        attention_weights = tf.matmul(inputs, self.attention_weights)
        attention_weights *= self.temperature
        attention_weights -= tf.math.reduce_max(  # numerical stability
            attention_weights, axis=-2, keepdims=True)
        attention_weights = tf.exp(attention_weights)
        attention_weights = self._attention_dropout(attention_weights)
        if mask is not None:
            attention_weights *= mask
        attention_weights /= (
            tf.reduce_sum(attention_weights, axis=-2, keepdims=True) + 1e-32)
        attention_weights = tf.expand_dims(attention_weights, -1)

        # compute multihead attention values
        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)
        values *= attention_weights
        values = tf.reduce_sum(values, axis=1, keepdims=True)
        values = tf.reshape(values, [-1, 1, self.final_dim])

        # compute aggregation
        values = tf.repeat(values, repeats=tf.shape(inputs)[1], axis=1)
        if self.merge_mode == "sum":
            aggregation = inputs + values
        else:
            aggregation = tf.concat([inputs, values], axis=-1)

        # feedforward 1
        results = tf.matmul(aggregation, self.feedforward_layer_1)
        results = self.activation(results + self.feedforward_layer_1_bias)

        # feedforward 2
        results = tf.matmul(results, self.feedforward_layer_2)
        results = results + self.feedforward_layer_2_bias
        results = self._feedforward_dropout(results)

        # combine and normalize
        results += inputs  # residual connection
        results = self.layer_norm(results)  # layer normalization

        # apply mask one last time
        if mask is not None:
            results *= mask
        return results

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
            "activation": activations.serialize(self.activation),
            "dropout": self.dropout,
            "merge_mode": self.merge_mode
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"SmallAttention": SmallAttention}


class PositionalEmbedding(layers.Layer):
    """
    Positional encoding layer that adds trainable positional embeddings
    to the input.
    Args:
        positional_activation: Activation function applied to the positional
                               embeddings.
    Input shape:
        3D tensor with shape `(batch_size, sequence_length, input_dim)`.
    Output shape:
        3D tensor with shape `(batch_size, sequence_length, input_dim)`.
    Attributes:
        positional_activation: Activation function applied to the positional
                               embeddings.
    Methods:
        call(inputs, mask=None): Performs the positional encoding on the input.
        get_config(): Returns the configuration of the layer.
        from_config(cls, config): Instantiates the layer from a configuration
                                  dictionary.
        get_custom_objects(): Returns a dictionary mapping the layer name to
                              the layer class.
    References:
        - `Attention Is All You Need`_
    .. _`Attention Is All You Need`: https://arxiv.org/abs/1706.03762
    """
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
        """
        Generates positional embeddings.
        Args:
            inputs: Input tensor of shape
                    `(batch_size, sequence_length, input_dim)`.
            mask: Mask tensor of shape `(batch_size, sequence_length)`.
        Returns:
            A tensor of the same shape as the input tensor with
            positional embeddings added.
        """
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
    """Positional encoding layer for adding positional embeddings to inputs.
    Args:
        positional_activation (str or callable): Activation function to use
                                                 for the positional embeddings.
            If `None`, no activation is applied. Default is `None`.
    Attributes:
        embedding (tf.keras.layers.Embedding): Embedding layer for positional
                                               indices.
    """
    def __init__(self, positional_activation=None, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.positional_activation = activations.get(positional_activation)

    def build(self, input_shape):
        """Builds the embedding layer for positional indices.
        Args:
            input_shape (tuple): Shape of input tensor.
        """
        seq_len, dim = input_shape[1], input_shape[2]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=seq_len, output_dim=dim)

    def call(self, inputs, mask=None):
        """Computes positional embeddings and adds them to inputs.
        Args:
            inputs (tf.Tensor): Input tensor to add positional embeddings to.
            mask (tf.Tensor, optional): Mask tensor for inputs.
                                        Default is `None`.
        Returns:
            tf.Tensor: Tensor with positional embeddings added to inputs.
        """
        input_length = tf.shape(inputs)[1]
        position_indices = tf.range(
            input_length, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.embedding(position_indices)
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


class SafeGlobalAveragePooling1D(GlobalPooling1D):
    def __init__(self, data_format="channels_last", **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        if mask is not None:
            mask = tf.cast(mask, inputs[0].dtype)
            mask = tf.expand_dims(
                mask, 2 if self.data_format == "channels_last" else 1
            )
            inputs *= mask
            denom = tf.reduce_sum(mask, axis=steps_axis, keepdims=self.keepdims) + 1e-30
            return K.sum(
                inputs, axis=steps_axis, keepdims=self.keepdims
            ) / denom
        else:
            return K.mean(inputs, axis=steps_axis, keepdims=self.keepdims) + 1e-30

    def compute_mask(self, inputs, mask=None):
        return None


class SmallDecoder(layers.Layer):
    def __init__(
        self,
        num_heads,
        intermediate_dim=None,
        activation="relu",
        dropout=0,
        name=None,
        merge_mode="sum",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = activations.get(activation)
        self.dropout = dropout
        self.merge_mode = merge_mode

        self._built = False
        self.supports_masking = True

    def _build(self, input_shape):
        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.intermediate_dim is None:
            self.intermediate_dim = in_dim // self.num_heads
        self.final_dim = self.num_heads * self.intermediate_dim

        # dropouts
        self._attention_dropout = layers.Dropout(rate=self.dropout)
        self._feedforward_dropout = layers.Dropout(rate=self.dropout)

        # attention mechanism
        dtype = tf.float32
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(in_dim, self.num_heads),
            initializer=clone_initializer("orthogonal"),
            dtype=dtype)
        self.temperature = self.add_weight(
            name="temperature", shape=(1, 1, self.num_heads),
            initializer="ones", dtype=dtype)

        initializer = "glorot_uniform"
        self.values = self.add_weight(
            name="values",
            shape=(in_dim, self.num_heads, self.intermediate_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)
        self.operator = self.add_weight(
            name="operator",
            shape=(self.final_dim, in_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)

        # feed forward layers
        initializer = "glorot_normal"
        ff_in_dim = in_dim if self.merge_mode == "sum" else self.final_dim + in_dim
        self.feedforward_layer_1 = self.add_weight(
            name="feedforward_layer_1",
            shape=(ff_in_dim, in_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)
        self.feedforward_layer_1_bias = self.add_weight(
            name="feedforward_layer_1_bias",
            shape=(1, 1, in_dim),
            initializer="zeros",
            dtype=dtype)
        self.feedforward_layer_2 = self.add_weight(
            name="feedforward_layer_2",
            shape=(in_dim, in_dim),
            initializer=clone_initializer(initializer),
            dtype=dtype)
        self.feedforward_layer_2_bias = self.add_weight(
            name="feedforward_layer_2_bias",
            shape=(1, 1, in_dim),
            initializer="zeros",
            dtype=dtype)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, axis=-1)

        self._built = True

    def call(self, inputs, mask=None):
        if not self._built:
            self._build(inputs.shape)

        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)

        # compute multihead attention weights
        attention_weights = tf.matmul(inputs, self.attention_weights)
        attention_weights *= self.temperature
        attention_weights -= tf.math.reduce_max(  # numerical stability
            attention_weights, axis=-2, keepdims=True)
        attention_weights = tf.exp(attention_weights)
        attention_weights = self._attention_dropout(attention_weights)
        if mask is not None:
            attention_weights *= mask
        attention_weights = tf.expand_dims(attention_weights, -1)

        # compute multihead attention values
        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)
        values *= attention_weights

        # compute cumulated sums and reshape
        attention_weights = tf.clip_by_value(
            tf.math.cumsum(attention_weights, axis=1), 1e-32, float("inf"))
        values = tf.math.cumsum(values, axis=1)
        values = tf.reshape(values, [-1, values.shape[1], self.final_dim])

        # compute aggregation
        if self.merge_mode == "sum":
            aggregation = inputs + values
        else:
            aggregation = tf.concat([inputs, values], axis=-1)

        # feedforward 1
        results = tf.matmul(aggregation, self.feedforward_layer_1)
        results = self.activation(results + self.feedforward_layer_1_bias)

        # feedforward 2
        results = tf.matmul(results, self.feedforward_layer_2)
        results = results + self.feedforward_layer_2_bias
        results = self._feedforward_dropout(results)

        # combine and normalize
        results += inputs  # residual connection
        results = self.layer_norm(results)  # layer normalization

        # apply mask one last time
        if mask is not None:
            results *= mask
        return results

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
            "activation": activations.serialize(self.activation),
            "dropout": self.dropout,
            "merge_mode": self.merge_mode
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
        self,
        num_heads,
        intermediate_dim,
        out_dim=None,
        activation="relu",
        dropout=0,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activations.get(activation)
        self.dropout = dropout

    def build(self, input_shape):
        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.out_dim is None:
            self.out_dim = in_dim

        self.flatten_dim = self.num_heads * self.intermediate_dim

        # dropouts
        self._attention_dropout = layers.Dropout(rate=self.dropout)
        self._feedforward_dropout = layers.Dropout(rate=self.dropout)

        # attention weights
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(in_dim, self.num_heads),
            initializer=clone_initializer("orthogonal"))
        self.values = self.add_weight(
            name="values",
            shape=(in_dim, self.num_heads, self.intermediate_dim),
            initializer=clone_initializer("glorot_normal"))
        self.temperature = self.add_weight(
            name="temperature",
            shape=(1, 1, self.num_heads), initializer="ones")

        # feed forward
        self.feedforward = self.add_weight(
            name="feedforward", shape=(self.flatten_dim, self.out_dim),
            initializer=clone_initializer("glorot_normal"))
        self.feedforward_bias = self.add_weight(
            name="feedforward_bias", shape=(1, self.out_dim),
            initializer="zeros")

    def call(self, inputs, mask=None):
        # compute attention weights
        attention_weights = tf.matmul(inputs, self.attention_weights)
        attention_weights *= self.temperature
        attention_weights = tf.exp(attention_weights)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            attention_weights *= mask
        attention_weights = self._attention_dropout(attention_weights)
        attention_weights /= (
            tf.reduce_sum(attention_weights, axis=-2, keepdims=True) + 1e-30)
        attention_weights = tf.expand_dims(attention_weights, -1)

        # compute multi head values
        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)
        values *= attention_weights
        values = tf.reduce_sum(values, axis=1)
        values = tf.reshape(values, [-1, self.flatten_dim])
        values = self.activation(values)

        # feedforward
        results = tf.matmul(values, self.feedforward)
        results += self.feedforward_bias
        results = self._feedforward_dropout(results)
        results = results
        return results

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
            "out_dim": self.out_dim,
            "activation": activations.serialize(self.activation),
            "dropout": self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"SmallPooling": SmallPooling}


class GLU(layers.Layer):
    def __init__(self, axis=-1):
        super(GLU, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        self.input_dim = input_shape[self.axis]
        self.output_dim = self.input_dim // 2

        self.double_layer = layers.Dense(2 * self.input_dim)
        self.gate_layer = layers.Dense(self.output_dim)
        self.relu_layer = layers.Dense(self.output_dim, activation='relu')

    def call(self, inputs):
        # Double the dimensionality of the input tensor
        x = self.double_layer(inputs)

        # Split the input tensor along the specified axis into two halves
        a, b = tf.split(x, num_or_size_splits=2, axis=self.axis)

        # Compute the gated linear units (GLU) activation
        g = self.gate_layer(b)
        h = self.relu_layer(b)
        outputs = g * h

        # Concatenate the input with the GLU output
        return tf.concat([a, outputs], axis=self.axis)