import tensorflow as tf


class CentralityEncoding(tf.keras.layers.Layer):
    """CentralityEncoding layer.
    The Centrality Encoding measures the importance of each node in the graph.
    Here, importance is determined by the degree centrality, i.e. the degree of
    each node. Degrees are projected to the hidden size of the model via learnt 
    embeddings.

    Args:
        max_degree: Maximum degree of the input graphs
        d_model: Size of the hidden dimension of the model
        
    Call arguments:
        distances: `Tensor` of shape `(B, N, N)`, where `B` is the batch_size and `N` is
        number of nodes. Entries of 1 (or -1) indicate direct neighbors and 
        contribute to the centrality encoding of that node. 
    Returns:
        centrality_encoding: The result of the computation, of shape `(B, N, D)`,
        where `N` is for the number of nodes and `D` is the hidden dimension size 
        of the model. 
    """

    def __init__(self, max_degree, d_model):
        super(CentralityEncoding, self).__init__()
        self.centr_embedding = tf.keras.layers.Embedding(max_degree, d_model)

    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), tf.float32) # abs --> connections to vnode count
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, tf.float32)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality)
        return centrality_encoding


class SpatialEncoding(tf.keras.layers.Layer):
    """SpatialEncoding layer.
    The Spatial Encoding assigns each pairwise distance between nodes a learnable scalar
    which serve as a bias term in the self-attention module.

    Args:
        d_sp_enc: Degree of the intermediate layer to learn scalars for each distance. Default: 16.
        activation: Activation function for the mapping from distances to bias terms. Default: 'relu'.
        
    Call arguments:
        distances: `Tensor` of shape `(B, N, N)`, where `B` is the batch_size and `N` is
        number of nodes. Entries represent the pairwise mininum distances of nodes in the
        graph. Special values should be assigned to nodes which are not connected and to
        connections to the virtual node. 
    Returns:
        spatial_encoding: The result of the computation, of shape `(B, N, N)`,
        where `N` is for the number of nodes. 
    """
    def __init__(self, d_sp_enc=16, activation='relu'):
        super(SpatialEncoding, self).__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(d_sp_enc, activation=activation)
        self.dense2 = tf.keras.layers.Dense(1, activation=activation)

    def call(self, distances):
        expanded_inputs = tf.expand_dims(distances, axis=-1)
        outputs = self.dense1(expanded_inputs)
        outputs = self.dense2(outputs)
        spatial_encoding = tf.squeeze(outputs, axis=-1)
        return spatial_encoding


def scaled_dot_product_attention(q, k, v, min_distance_matrix, spatial_encoding, num_heads, mask=None, d_sp_enc=16, sp_enc_activation='relu'):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask must be broadcastable for addition. Spatial encoding bias is added
    to the query-key product matrix.

    Args:
        q: query shape == (..., seq_len_q, hidden_dim)
        k: key shape == (..., seq_len_k, hidden_dim)
        v: value shape == (..., seq_len_v, hidden_dim)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
        min_distance_matrix: Float tensor of pairwise minimum distances, 
            shape == (..., seq_len, seq_len)
        d_sp_enc: Dimension of hidden layer to learn spatial encoding
        sp_enc_activation: Keras activation function for spatial encoding

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Spatial encodings
    spatial_encoding_bias = spatial_encoding(min_distance_matrix)
    spatial_encoding_bias = tf.expand_dims(spatial_encoding_bias, axis=1)
    spatial_encoding_bias = tf.repeat(spatial_encoding_bias, repeats=num_heads, axis=1)
    scaled_attention_logits += spatial_encoding_bias

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention layer.
    The implementation largely follows the implementation of the original transformer
    architecture with the addition of spatial encodings.

    Args:
        d_model: Size of the hidden dimension of the model
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        d_sp_enc: Dimension of hidden layer to learn spatial encoding.
        sp_enc_activation: Keras activation function for spatial encoding.
        
    Call arguments:
        q: query shape == (..., seq_len_q, hidden_dim)
        k: key shape == (..., seq_len_k, hidden_dim)
        v: value shape == (..., seq_len_v, hidden_dim)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
        min_distance_matrix: Float tensor of pairwise minimum distances, 
            shape == (..., seq_len, seq_len)

    Returns:
        output, attention_weights
    """
  
    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, min_distance_matrix, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, min_distance_matrix, self.spatial_encoding, self.num_heads, mask, self.d_sp_enc, self.sp_enc_activation)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """Feedforward Network.

    Args:
        d_model: Size of the hidden dimension of the model.
        dff: Intermediate size of the model.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class GraphormerBlock(tf.keras.layers.Layer):
    """Graphormer layer.
    Corresponds to a transformer block. Pre-layer normalization is used in 
    accordance with the paper, i.e. before the attention module and the 
    feed-forwards network respectively.

    Args:
        d_model: Size of the hidden dimension of the model.
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        rate: Dropout rate. Default: 0.1.
        d_sp_enc: Dimension of hidden layer to learn spatial encoding. Default: 128.
        sp_enc_activation: Keras activation function for spatial encoding. Default: 'relu'
        
    Call arguments:
        training: Bool indicating if training or inference.
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
        min_distance_matrix: Float tensor of pairwise minimum distances, 
            shape == (..., seq_len, seq_len)
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=128, sp_enc_activation='relu'):
        super(GraphormerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, min_distance_matrix):

        residual = x
        x_norm = self.layernorm1(x) # NEW: Pre-LayerNorm
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, min_distance_matrix, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = residual + attn_output  # (batch_size, input_seq_len, d_model)

        residual = out1
        out1_norm = self.layernorm2(out1) # NEW: Pre-LayerNorm
        ffn_output = self.ffn(out1_norm)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = residual + ffn_output  # (batch_size, input_seq_len, d_model)

        return out2