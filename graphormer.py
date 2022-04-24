import tensorflow as tf
from layers import CentralityEncoding, GraphormerBlock

class Graphormer(tf.keras.Model):
    """Graphormer model.
    Implements a Graphormer model following Ying et al. (2021). *Do Transformers Really
    Perform Bad for Graph Representation?* arXiv:2106.05234. Inputs represent undirected
    graphs. Inputs all need to be padded to equal lengths, node_id 0 is assumed as the
    padding token.

    Args:
        num_layers: Number of Graphormer blocks.
        d_model: Size of the hidden dimension of the model.
        num_heads: Number of attention heads. d_model must be divisible by num_heads.
        input_vocab_size: Maximum number of unique node ids across all inputs, should be 
        set large enough.
        dropout: Dropout rate in the Graphormer blocks. Default: 0.1.
        dff: Intermediate size of the feedforward networks in the Graphormer blocks. Default: 512.
        d_sp_enc: Dimension of hidden layer to learn spatial encoding. Default: 128.
        sp_enc_activation: Keras activation function for spatial encoding. Default: 'relu'.
        max_num_nodes: Maximum number of nodes per graph. Default: 256.
        max_degree: Maximum degree per graph. Default: 256.
        model_head: one of 'VNode', 'average'. 'average' pools the outputs over all positions, 
        'VNode' only selects the learnt output of the first position (assumes a virtual node
        as the first token, e.g. similar as in BERT)
        top_dropout: Dropout rate in the classification head. Default: 0.1.
        d_top: Hidden dimension of the classification head. Default: 64.
        top_activation: Activation function in the classification head. Default: 'relu'.
        initial_bias: Initial bias of the classifier. Default: 0.
        concat_n_layers: Int, Number of Graphormer layers of which outputs to concatenate to
        feed into the classification head. Default: 1.
        
    Call arguments:
        inputs: tuple/list of node_features, distance_matrix
            node_features: (batch_size, max_num_nodes) tensor of node features
            distance_matrix: (batch_size, max_num_nodes, max_num_nodes) pairwise minimum distances
            between nodes
    Returns:
        centrality_encoding: The result of the computation, of shape `(B, N, D)`,
        where `N` is for the number of nodes and `D` is the hidden dimension size 
        of the model. 
    """
  
    def __init__(self, num_layers, d_model, num_heads, input_vocab_size,
                dropout=0.1, dff=512, d_sp_enc=128, sp_enc_activation='relu', 
                max_num_nodes=256, max_degree=256, model_head='VNode', top_dropout=0.1, d_top=64,
                top_activation='relu', initial_bias=0, concat_n_layers=1):
        super(Graphormer, self).__init__()

        self.d_model = d_model
        self.max_num_nodes = max_num_nodes
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.centr_encoding = CentralityEncoding(max_degree, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.graphormer_layers = [GraphormerBlock(d_model, num_heads, dff, dropout, d_sp_enc, sp_enc_activation)
                        for _ in range(num_layers)]

        self.concat_n_layers = concat_n_layers
        self.model_head = model_head
        self.top_dropout = tf.keras.layers.Dropout(top_dropout)
        self.d_top = d_top
        self.top_activation = top_activation
        self.top_dense = tf.keras.layers.Dense(self.d_top, activation=self.top_activation, name='top_dense_1')
        self.output_bias = tf.keras.initializers.Constant(initial_bias)
        self.head = tf.keras.layers.Dense(1, activation='sigmoid', name='binary_classifier' 
                                    ,bias_initializer=self.output_bias)
        
    def create_padding_mask(self, nodes):
        return tf.cast(tf.math.equal(nodes, 0), tf.float32)

    def call(self, inputs, training):
        # All inputs passed in the first argument (tupel or list)
        node_features, distance_matrix = inputs

        # Generate masks
        mask = self.create_padding_mask(node_features)
        # mask transformation to add extra dimensions to add the padding to the attention logits.
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


        # Embedding
        embed = self.embedding(node_features)  # (batch_size, input_seq_len, d_model)
        embed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embed += self.centr_encoding(distance_matrix) 
        out = self.dropout(embed, training=training)

        hidden_states = [] # List to hold hidden states per graphormer block

        # Graphormer layers
        for i in range(self.num_layers):
            out = self.graphormer_layers[i](out, training, attention_mask, distance_matrix)  # (batch_size, inp_seq_len, d_model)
        if self.concat_n_layers > 1:
            hidden_states.append(out)

        # Model head
        if self.model_head not in ['VNode', 'average']:
            raise NameError('model_head must be in ["VNode", "average"]')

        # Concatenate hidden states
        if self.concat_n_layers > 1:
            out = tf.keras.layers.Concatenate()(tuple([hidden_states[i] for i in range(-self.concat_n_layers, 0)]))
        
        # Virtual Node
        if self.model_head == 'VNode':
            out = out[:, 0, :]

        # Average Pooling
        if self.model_head == 'average':
            out = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(out, mask)

        out = self.top_dropout(out)
        out = self.top_dense(out)

        out = self.head(out)

        return out

test = Graphormer(2, 512, 4, 100)
print(test)