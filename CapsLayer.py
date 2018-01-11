import numpy as np
import tensorflow as tf

from config import cfg

epsilon = 1e-9


class CapsLayer(object):
    """
    Args:
    -----
        input: 4-D tensor with shape ()
        num_outputs: the number of the output vectors of a capsule
        vector_length: the length of the output vector of a capsule
        layer_type: string, 'FC' or 'CONV'
        with_routing: boolean, this capsule is routing with the lower-level layer capsule

    Return:
    -------
        A 4-D Tensor
    """

    def __init__(self, num_outputs, vector_length, layer_type, with_routing):
        self.num_outputs = num_outputs
        self.vector_length = vector_length
        self.layer_type = layer_type
        self.with_routing = with_routing

    def __call__(self, input, kernel_size=None, stride=None):
        """
        if layer_type == 'CONV':
            'kernel_size' and 'stride' will be set
        """

        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # the PrimaryCaps layer, no routing with the previous layer
                # input: [batch_size, 20, 20, 256]
                assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

                # Note: using "ReLU" non-linear function before applying the "squashing" function
                # Each capsule i: [batch_size, 6, 6, 32]   6*6*32==1152
                capsules = tf.contrib.layers.conv2d(input,
                                                    self.num_outputs * self.vector_length,
                                                    self.kernel_size,
                                                    self.stride,
                                                    padding='VALID',
                                                    activation_fn=tf.nn.relu)
                capsules.reshape(capsules, (cfg.batch_size, -1, self.vector_length, 1))   # [batch_size, 6*6*32=1152, 8, 1]
                capsules = squash(capsules)

                assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                return capsules

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer and do routing with PrimaryCaps
                # Reshape the input(i.e. the output vector u_i from previous layer) into [batch_size, 1152, 1, 8, 1] (batch_size, num_caps, num_digit_caps, )
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, self.vector_length, 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_(l+1), 1, 1]
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return capsules

def routing(input, b_IJ):
    """ Dynamic Routing Algorithm -- Routing by Agreement
    Args:
    -----
        input: A Tensor with shape [batch_size, num_caps_l, 1, length_u_i(8), 1], where num_caps_l means the number of capsules in layer l
        b_IJ : raw coefficient with shape [batch_size, 1152, 10, 1, 1]
    Return:
    -------
        A Tensor of shape [batch_size, num_caps_(l+1), length_v_j(16), 1]
        the vector output v_j in the layer l+1

    Note:
        u_i represents the vector output of capsule i in the layer l
        v_j represents the vecotr output of capsule j in the layer l+1
    """

    """ Compute the prediction vectors u_j|i = W * u_i  (Eq.2) """
    # W: [1, num_caps_i, num_caps_j, length_u_i, length_v_j]
    # add 1 dim in the beginning to make it easy to tile this array for each instance in the batch
    W = tf.get_variable('weight', shape=(1, 1152, 10, 8, 16),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))

    # For computational efficiency, we do tiling for input and W before tf.matmul()
    # input size: [batch_size, 1152, 1, 8, 1] ===> [batch_size, 1152, 10, 8, 1]
    # W shape   : [1, 1152, 10, 8, 16]        ===> [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.file(W, [cfg.batch_size, 1, 1, 1, 1])

    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1]
    assert W.get_shape() == [cfg.batch_size, 1152, 10, 8, 16]

    # Compute the prediction vectors
    # For last 2 dims: [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    u_ji = tf.matmul(W, input, transpose_a=True)
    assert u_ji.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward,  u_ji_stopped = u_ji
    # In backward, no gradient passed back from u_ji_stopped to u_ji
    u_ji_stopped = tf.stop_gradient(u_ji, name='stop_gradient')


    # Implementation of Routing-by-Algorithm
    for r_iter in xrange(cfg.iter_routing):
        with tf.variable_scope('iter_%d' % r_iter):
            # the coupling coefficients c_IJ = [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            if r_iter == cfg.iter_routing - 1:
                # last iteration, use `u_ji` in order to receive gradients from the following graph
                # weighting u_ji with c_IJ, element-wise in the last 2 dims
                # the output capsule = [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_ji)
                # sum all the capsules ==> [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axim=1, keep_dims=True)
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                v_J = squash(s_J)
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:
                # inner iteration, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_ji_stopped)  # x * y element-wise
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # reshape & tile v_J from [batch_size, 1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # Then, , matmul in the last two dim: [16, 1].T x [16, 1] => [1, 1],
                # reduce mean in the batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                agreement = tf.matmul(u_ji_stopped, v_J_tiled, transpose_a=True, name='agreement')
                assert agreement.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ = tf.reduce_sum(agreement, axis=0, keep_dims=True)
                b_IJ += agreement

    return v_J


def squash(vector):
    """ Squashing function corresponding to Eq. 1
    Args:
    -----
        vector: A tensor with shape [batch_size, 1, num_caps(10), vector_length(16), 1] or [batch_size, num_caps(10), vector_length(16), 1].

    Return:
    -------
        A tensor with the same shape as vector but squashed in 'vector_length' dimension.
    """

    # If the norm of a vector is 0 ==> The gradient will be undifined N/A
    # ==> To prevent it, ||s|| := sqrt(sum(si^2) + epsilon)
    vector_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)

    scalar_factor = vector_squared_norm / (1 + vector_squared_norm) / tf.sqrt(vector_squared_norm + epsilon)
    squashed_vector = scalar_factor * vector  # element-wise

    return(squashed_vector)
