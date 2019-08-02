# -*- coding: utf-8 -*-
#/usr/bin/python3

from bert.bert_model_for_PAGE import get_shape_list

import numpy as np
import tensorflow as tf

def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(seq_k, seq_q, Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 num_heads=8,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(seq_k, seq_q, outputs, num_heads,type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(seq_k, seq_q, outputs, num_heads,type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)

        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        outputs = mask(seq_k, seq_q, outputs,num_heads, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def mask(seq_k, seq_q, inputs,num_heads, type=None):

    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = get_attn_key_pad_mask(seq_k=seq_k, seq_q=seq_q, PAD_ID=0)
        # Apply masks to inputs
        masks = tf.tile(masks,[num_heads, 1, 1])
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = get_non_pad_mask(seq_k=seq_k, seq_q=seq_q, PAD_ID=0)
        masks = tf.tile(masks, [num_heads, 1, 1])
        # Apply masks to inputs
        outputs = inputs*masks

    elif type in ("f", "future", "right"):
        masks = get_subsequent_mask(inputs)
        # masks = tf.tile(masks, [num_heads, 1, 1])

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs






def get_non_pad_mask(seq_k, seq_q, PAD_ID):
    '''

    :param seq: [batch_size,len]
    :return:
    '''
    '''
    masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    '''
    masks = tf.cast(tf.math.not_equal(seq_q, PAD_ID), tf.float32)
    masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    masks = tf.tile(masks, [1, 1, get_shape_list(seq_k)[1]])  # (N, T_q, T_k)
    return masks


def get_attn_key_pad_mask(seq_k,seq_q,PAD_ID):
    ''' For masking out the padding part of key sequence. '''
    # print(get_shape_list(seq_k,expected_rank=2))

    padding_mask = tf.cast(tf.math.not_equal(seq_k,PAD_ID),tf.float32)
    masks = tf.expand_dims(padding_mask, 1)  # (N, 1, T_k)
    masks = tf.tile(masks, [1, get_shape_list(seq_q,expected_rank=2)[1], 1])  # (N, T_q, T_k)
    # print(get_shape_list(masks,expected_rank=3))
    return masks




def get_subsequent_mask(inputs):
    ''' For masking out the subsequent info. '''

    diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()         # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(tril, 0), [get_shape_list(inputs)[0], 1, 1])  # (N, T_q, T_k)

    return masks




def multihead_attention(seq_k,
                        seq_q,
                        queries, keys, values,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=False) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False) # (N, T_k, d_model)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(seq_k, seq_q, Q_, K_, V_, causality, dropout_rate, training,num_heads)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
          3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)