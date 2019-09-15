import tensorflow as tf
'''
regression_model
'''
def constant(x):
    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    Parameters
    ----------
    x : tensor
        An tensor with shape (1, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : tensor
        An array with shape (1, p) with the values of the regression
        model.
    """
    return tf.constant(1, shape=[1, 1], dtype=tf.float64)

def linear(x):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x : tensor
        A tensor with shape (1, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : tensor
        A tensor with shape (1, p) with the values of the regression
        model.
    """
    x = tf.reshape(x,[1,-1])
    return tf.concat([tf.constant(1, shape=[1, 1], dtype=tf.float64), x], 1)