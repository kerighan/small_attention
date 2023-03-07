from tensorflow.keras import layers, activations, initializers
import tensorflow as tf
import numpy as np


def clone_initializer(initializer):
    if not isinstance(initializer, initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


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
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = activations.get(activation)
        self.dropout = dropout

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
            name="temperature",
            shape=(1, 1, self.num_heads),
            initializer="ones",
            dtype=dtype)

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
        self.feedforward_layer_1 = self.add_weight(
            name="feedforward_layer_1",
            shape=(in_dim, in_dim),
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

        # compute multihead attention weights
        attention_weights = tf.matmul(inputs, self.attention_weights)
        attention_weights *= self.temperature
        attention_weights -= tf.math.reduce_max(  # numerical stability
            attention_weights, axis=-2, keepdims=True)
        attention_weights = tf.exp(attention_weights)
        attention_weights = self._attention_dropout(attention_weights)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            attention_weights *= mask
        attention_weights /= (
            tf.reduce_sum(attention_weights, axis=-2, keepdims=True) + 1e-32)
        attention_weights = tf.expand_dims(attention_weights, -1)

        # compute multihead attention values
        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)
        values *= attention_weights
        values = tf.reduce_sum(values, axis=1, keepdims=False)
        values = tf.reshape(values, [-1, self.final_dim])

        # compute aggregation
        operator = tf.matmul(values, self.operator)
        operator = operator[:, None, :]
        operator = tf.repeat(operator, repeats=tf.shape(inputs)[1], axis=1)
        aggregation = inputs + operator

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
            "dropout": self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {"SmallAttention": SmallAttention}


class SmallPooling(layers.Layer):
    def __init__(
        self,
        num_heads,
        out_dim=None,
        activation=None,
        dropout=0,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.activation = activations.get(activation)
        self.dropout = dropout

    def build(self, input_shape):
        seq_len, in_dim = input_shape[-2:]
        self.seq_len = seq_len
        self.in_dim = in_dim
        if self.out_dim is None:
            self.out_dim = in_dim

        intermediate_dim = self.out_dim // self.num_heads
        self.flatten_dim = self.num_heads * intermediate_dim

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
            shape=(in_dim, self.num_heads, intermediate_dim),
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

        # feedforward
        results = tf.matmul(values, self.feedforward)
        results += self.feedforward_bias
        results = self._feedforward_dropout(results)
        results = self.activation(results)
        return results

    def compute_mask(self, _, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
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
        attention_initializer = "orthogonal"
        self.attention_weights = self.add_weight(
            name="attention_weights", shape=(in_dim, self.num_heads),
            initializer=clone_initializer(attention_initializer))
        self.temperature = self.add_weight(
            shape=(1, 1, self.num_heads),
            initializer="ones", name="temperature")

        operator_initializer = "glorot_uniform"
        self.values = self.add_weight(
            shape=(in_dim, self.num_heads, self.intermediate_dim),
            initializer=clone_initializer(operator_initializer),
            name="values")
        self.operator = self.add_weight(
            name="operator", shape=(self.final_dim, vector_dim),
            initializer=clone_initializer(operator_initializer))

        # feed forward layers
        self.feedforward_layer_1 = self.add_weight(
            name="feedforward_layer_1", shape=(vector_dim + in_dim, in_dim),
            initializer=clone_initializer("glorot_uniform"))
        self.feedforward_layer_1_bias = self.add_weight(
            name="feedforward_layer_1_bias", shape=(1, 1, in_dim),
            initializer="zeros")
        self.feedforward_layer_2 = self.add_weight(
            name="feedforward_layer_2", shape=(in_dim, in_dim),
            initializer=clone_initializer("glorot_uniform"))
        self.feedforward_layer_2_bias = self.add_weight(
            name="feedforward_layer_2_bias",
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
        attention_weights = tf.matmul(inputs, self.attention_weights)
        attention_weights -= tf.math.reduce_max(attention_weights, axis=-2, keepdims=True)
        attention_weights *= self.temperature
        attention_weights = tf.exp(attention_weights)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            attention_weights *= mask
        attention_weights = tf.repeat(
            attention_weights[:, :, None, :], repeats=self.seq_len, axis=2)
        attention_weights *= self.causal_mask
        attention_weights /= (
            tf.reduce_sum(attention_weights, axis=-2, keepdims=True) + 1e-30)
        attention_weights = tf.expand_dims(attention_weights, -1)

        values = tf.einsum("bsi,ijk->bsjk", inputs, self.values)[:, None]
        values *= attention_weights
        values = tf.reduce_sum(values, axis=2, keepdims=False)
        values = tf.reshape(values, [-1, self.seq_len, self.final_dim])
        operator = tf.matmul(values, self.operator)

        if len(vector.shape) == 2:
            vector = vector[:, None, :]

        operator += vector
        concat = tf.concat([inputs, operator], axis=-1)

        # feedforward 1
        res = tf.matmul(concat, self.feedforward_layer_1)
        res = self.activation(res + self.feedforward_layer_1_bias)

        # feedforward 2
        res = tf.matmul(res, self.feedforward_layer_2)
        res = self.activation(res + self.feedforward_layer_2_bias)

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
