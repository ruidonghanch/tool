from scipy.stats import norm
import numpy as np
import math
import tensorflow as tf
'''
定义normal_cdf_tf激活函数
normal_cdf_tf返回一个正态分布的cdf，其梯度为正态分布的pdf
'''
def normal_cdf(x):
    return norm.cdf(x)


def normal_cdf_gradient(x):
    return (1 / math.sqrt(2. * math.pi)) * math.exp(- (x * x) / 2.)


normal_cdf_np = np.vectorize(normal_cdf)
normal_cdf_gradient_np = np.vectorize(normal_cdf_gradient)

normal_cdf_np_64 = lambda x: normal_cdf_np(x).astype(np.float64)
normal_cdf_gradient_np_64 = lambda x: normal_cdf_gradient_np(x).astype(np.float64)


def normal_cdf(x):
    return norm.cdf(x)


np_normal_cdf = np.vectorize(normal_cdf)


def d_normal_cdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(- (x * x) / 2)


np_d_normal_cdf = np.vectorize(d_normal_cdf)

np_d_normal_cdf_64 = lambda x: np_d_normal_cdf(x).astype(np.float64)


def tf_d_normal_cdf(x, name=None):
    with tf.name_scope(name, "d_normal_cdf", [x]) as name:
        y = tf.py_func(np_d_normal_cdf_64,
                       [x],
                       [tf.float64],
                       name=name,
                       stateful=False)
        return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def normal_cdf_grad(op, grad):
    x = op.inputs[0]

    n_gr = tf_d_normal_cdf(x)
    return grad * n_gr


np_normal_cdf_64 = lambda x: np_normal_cdf(x).astype(np.float64)


def tf_normal_cdf(x, name=None):
    with tf.name_scope(name, "normal_cdf", [x]) as name:
        y = py_func(np_normal_cdf_64,
                    [x],
                    [tf.float64],
                    name=name,
                    grad=normal_cdf_grad)  # <-- here's the call to the gradient
        return y[0]
