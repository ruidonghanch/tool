import tensorflow as tf
from scipy.stats import norm
import numpy as np
from sklearn import gaussian_process
from tensorflow.python.framework import ops
import math
import random
from normal_activation import tf_normal_cdf
from regr_model import *
from select_model import *


'''
created by statruidong 20190915
version 0.1
environment:
    sklearn 0.19.2
    tensorflow 1.13.0
'''

tf.reset_default_graph()
# 学习参数
DECAY_STEP = 300
DECAY_RATE = 0.96

ZERO_THRESHOLD = tf.constant(1e-1, shape=[1, 1], dtype=tf.float64)  # 数值计算的绝对值小于该数时，被强制转换为0
regr_dict = {'constant': constant, 'linear': linear}


'''
criterion_model
'''
def gp_ucb_calculate(mu, sigma, k):
    k = tf.constant(k,name='k',dtype=tf.float64)
    return tf.add(mu,tf.multiply(sigma,k))


def ei_calculate(mu, sigma, best_point_value, xi):
    '''
    计算采样点为x时,按照EI准则算出的目标函数值
    EI准则公式参考https://www.cnblogs.com/statruidong/p/10558089.html

    parameters
    ----------
    mu:tensor,尺寸为(1,1),代表某采样点的预测mu
    sigma:tensor,尺寸为(1,1),代表某采样点的预测sigma
    best_point_value:tensor,尺寸为(1,1),代表历史最佳采样点相应函数值
    xi:平衡探索与利用，并且xi>=0

    returns
    -------
    y:tesnor,尺寸为(1,1),代表该采样点EI值
    '''

    # check parameter
    if xi < 0:
        raise ValueError("The xi value must satisfy xi >= 0.")
    else:
        xi = tf.constant(xi, shape=[1, 1], dtype=tf.float64)
    if best_point_value is None:
        raise ValueError("best_point_value is empty")
    else:
        best_point_value = tf.constant(best_point_value, shape=[1, 1], dtype=tf.float64)

    if tf.less(sigma[0, 0], ZERO_THRESHOLD[0, 0]) is True:
        print('value error')
    Z = tf.truediv(tf.subtract(mu, best_point_value), sigma)

    part1 = tf.multiply(tf.subtract(mu, tf.add(best_point_value, xi)), tf_normal_cdf(Z))

    var1 = tf.exp(tf.multiply(tf.constant(-0.5, dtype=tf.float64), tf.square(Z)))
    var2 = tf.constant(1 / np.sqrt(2 * np.pi), dtype=tf.float64)
    var3 = tf.multiply(var2, var1)
    part2 = tf.multiply(sigma, var3)

    return tf.add(part1,part2)


# 建立高斯过程回归模型
def fit(X, y, regr_model,initial_theta,adaptive):
    MACHINE_EPSILON = np.finfo(np.double).eps
    #注意theta初始化范围，若theta最小值太小，可能造成估计方差太大
    #将random_state调成10，从而提高使用cobyla对theta估计的准确性
    if adaptive == True:
        thetaL = 0.5*initial_theta
        thetaU = 2*initial_theta
    else:
        thetaL = 1e-8*np.ones([1,DIMESSION],dtype=float)
        thetaU = 100*np.ones([1,DIMESSION],dtype=float)
    gp = gaussian_process.GaussianProcess(regr=regr_model, normalize=True,
                                          theta0= initial_theta,
                                          thetaL= thetaL,
                                          thetaU= thetaU,
                                          nugget=50 * MACHINE_EPSILON,random_start=25)  # 不能使用Welch
    gp.fit(X, y)
    return gp



'''
具体计算函数
'''
def calculate(x, gp, regr_model):
    '''
    计算采样点为x时的预测方差
    x -- > sigma(x)

    parameters
    ----------
    x:tesnor,一个尺寸为(1,n_features)的采样点
    gp:GaussianProcess对象

    returns
    -------
    mu:tensor,尺寸为(1,1),计算出的估计均值
    sigma:tensor,尺寸为(1,1)计算出的估计方差值
    '''
    reduced_likelihood_function_value, par = gp.reduced_likelihood_function()
    C = tf.constant(par['C'], dtype=tf.float64)
    Ft = tf.constant(par['Ft'], dtype=tf.float64)
    G = tf.constant(par['G'], dtype=tf.float64)
    # print(gp.regr(x))
    # regr constant
    f = regr_dict[regr_model](x)
    beta = tf.constant(gp.beta, dtype=tf.float64)
    gamma = tf.constant(gp.gamma, dtype=tf.float64)
    sigma2 = tf.constant(gp.sigma2, shape=[1, 1], dtype=tf.float64)

    # Normalize input
    X_mean = tf.constant(gp.X_mean, dtype=tf.float64)
    X_std = tf.constant(gp.X_std, dtype=tf.float64)
    # reshape
    X_mean = tf.reshape(X_mean, [1, -1])
    X_std = tf.reshape(X_std, [1, -1])

    x = tf.truediv(tf.subtract(x, X_mean), X_std)

    X = tf.constant(gp.X, dtype=tf.float64)  # X是经过标准化操作后的X
    theta_ = tf.constant(gp.theta_, dtype=tf.float64, shape=[DIMESSION, 1])

    l1_distance = tf.subtract(X, tf.matmul(tf.constant(1, shape=[X.shape[0], 1], dtype=tf.float64), x))
    l2_distance = tf.square(l1_distance)
    r = tf.exp(-tf.matmul(l2_distance, theta_))  # shape(X.shape[0],1)

    rt = tf.matrix_triangular_solve(C, r, lower=True)

    p = tf.subtract(tf.matmul(tf.transpose(Ft, [1, 0]), rt), tf.transpose(f, [1, 0]))  # p.shape=[feature+1,1]
    u = tf.matrix_triangular_solve(tf.transpose(G, [1, 0]), p, lower=True)

    mu = tf.matmul(f, beta) + tf.matmul(tf.reshape(r, [1, -1]), gamma)

    MSE = tf.matmul(sigma2, tf.subtract(tf.constant(1, shape=[1, 1], dtype=tf.float64),
                                        tf.subtract(tf.reduce_sum(tf.square(rt)), tf.reduce_sum(tf.square(u)))))
    sigma = tf.sqrt(MSE)
    return mu, sigma, sigma2

'''
目标优化器
'''
def train_step(criterion, global_step, initial_learning_rate):
    # 注意原始目标是最大化准则
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=DECAY_STEP,
                                               decay_rate=DECAY_RATE, staircase=True)
    loss = tf.multiply(tf.constant(-1, dtype=tf.float64), criterion)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # 注意有符号，原始目标是最大化ei
    return optimizer


def get_criterion(sample_point,X,y,mu,sigma, regr_model,criterion_model,xi,k):
    best_point_value = np.max(y)

    if criterion_model == 'gp_ucb':
        return gp_ucb_calculate(mu, sigma, k)
    else:
        return ei_calculate(mu, sigma, best_point_value, xi)

def get_next_point(X, y,
                   iter_num = 5000,
                   edgeU = None,
                   edgeL = None,
                   use_n = 10,
                   regr_model='constant',
                   criterion_model='gp_ucb',
                   xi=0.01,
                   k=None,
                   lr=0.0001,
                   min_iter=1000,
                   criterion_eps=1e-4,
                   initial_k=2,
                   decay_rate=0.9,
                   decay_num=40,
                   initial_theta=None):
    '''
       parameters
       ----------
       X:np.array,尺寸为(n_sample,n_attribute),代表采样点
       y:np.array,尺寸为(n_sample,1)

       edgeL:array,尺寸为(1,n_attribute),采样上界
       degeU:array,尺寸为(1,n_attribute)，采样下界
       iter_num:int,代表最大迭代次数

       use_n:int,default 10 采用多个点并行搜索
       regr_model:'linear' or 'contanst'
       criterion_model:'gp_ucb' or 'ei'

       xi:float,default 0.01，仅对ei有用，平衡探索与利用

       k:float,default None,gp_ucb参数,如果设置None表示自动调节k
       initial_k:float,default 2,仅当k为None时生效,gp_ucb参数
       decay_rate:float,default 0.9,仅当k为None时生效,gp_ucb参数
       decay_num:int,default 40,仅当k为None时生效,gp_ucb参数
                 k = initial_k*(decay_rate)**(n_sample/dacay_num)

       lr:float,学习率
       min_iter:int,default 1000,最低迭代次数
       criterion_eps:float, default 1e-4，1000次迭代后若criterion值小于该阈值，迭代停止
       initial_theta:np.array,尺寸为(1,n_attribute),表示核函数参数theta估计的初始值



       returns
       -------
       x:tesnor,尺寸为(1,n_attribute),代表下一个最佳采样点
       gp_theta: 本轮gp参数theta的估计值
    '''
    random_start = use_n#表示每一轮并行计算使用的采样点个数
    global DIMESSION
    DIMESSION = X.shape[1]
    n_sample = X.shape[0]

    # 记录迭代次数
    # 定义best_point
    global_step = tf.placeholder(tf.int32)
    # 标准化y
    y = y.reshape([-1,1])
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y = (y - y_mean) / y_std

    if edgeL is None and edgeU is None:
        scale = np.max(X, 0) - np.min(X, 0)
        edgeL = np.min(X,0) - scale * 0.2
        edgeU = np.max(X,0) + scale * 0.2

    else:
        edgeL = np.array(edgeL)
        edgeU = np.array(edgeU)


    # 进行并行采样
    initial_point = []
    for j in range(use_n):
        initial_point.append(rank_n_select(X, y, edgeL, edgeU))

    initial_point = tf.Variable(initial_point,dtype=tf.float64)

    #记录random_start已搜索的个数
    search_num = tf.constant(0,dtype=tf.int32)
    random_start = tf.constant(random_start,dtype=tf.int32)

    total_criterion = tf.constant(0,shape=[1,1],dtype=tf.float64)
    best_sample_point = tf.constant(np.ones([1,DIMESSION])*(-999),shape=[1,DIMESSION],dtype=tf.float64) #记录当前迭代轮数最佳的预估采样位置，初始化为每个数字均为-999
    best_criterion =  tf.constant( -1e10,shape=[1,1],dtype=tf.float64)#记录当前迭代轮数最佳准则值，-1e10

    #求解k
    if k is None:

        # 下面两行代码是按照Srinivas(2010)理论进行的，发现效果不理想
        # tao = 2*np.log((n_sample**(DIMESSION/2+2)*np.pi**2)/(3*np.sqrt(gp.sigma2)))
        # k = np.sqrt(tao)

        k = initial_k*(decay_rate)**(n_sample/decay_num)

    if initial_theta is None:
        initial_theta = 0.1 * np.ones((1, DIMESSION), dtype=float)
        adaptive = False
    else:
        initial_theta = np.array(initial_theta)
        adaptive = True

    gp = fit(X, y, regr_model, initial_theta, adaptive)

    #tensorflow循环体
    def cond(initial_point,search_num,random_start,total_criterion,best_criterion,best_sample_poin):
        return search_num < random_start
    def body(initial_point,search_num,random_start,total_criterion,best_criterion,best_sample_point):

        #均匀分布选取采样点
        sample_point = initial_point[search_num,:]

        #sample_point = tf.Variable(select_point,'sample_point',dtype=tf.float64)
        sample_point = tf.reshape(sample_point,shape=[1,DIMESSION])

        # penalty = tf.multiply(tf.constant(2,dtype=tf.float64),tf.reduce_sum(tf.square(tf.subtract(sample_point,X_best))))
        mu,sigma,sgima2 = calculate(sample_point, gp, regr_model)
        criterion = get_criterion(sample_point, X, y, mu, sigma, regr_model, criterion_model, xi, k)
        criterion = tf.reshape(criterion,shape=[1,1])
        # criterion -= penalty

        total_criterion += criterion
        search_num += 1

        #更新最佳位置
        best_sample_point = tf.cond(best_criterion[0,0]<criterion[0,0],lambda: sample_point,lambda: best_sample_point)
        best_criterion = tf.cond(best_criterion[0,0]<criterion[0,0],lambda: criterion,lambda: best_criterion)

        return initial_point,search_num,random_start,total_criterion,best_criterion,best_sample_point


    initial_point,search_num, random_start, total_criterion,best_criterion,best_sample_point = \
        tf.while_loop(cond,body,[initial_point,search_num,random_start,total_criterion,best_criterion,best_sample_point])

    #search_num = tf.constant(0,dtype=tf.float64)
    optimizer = train_step(total_criterion, global_step, lr)

    mu, sigma, sigma2 = calculate(best_sample_point, gp, regr_model)
    gp_theta = gp.theta_
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_criterion_prev = None
        beyond_edge = False #记录是否最佳值飞跃边界
        for i in range(iter_num):
            sess.run(optimizer, feed_dict={global_step: i})

            best_point_now = sess.run(best_sample_point)
            #print(best_point_now)
            if np.sum(np.array(best_point_now[0])<np.array(edgeU)) != DIMESSION:
                #continue
                #print('beyond the edge,please extend the edgeU')
                beyond_edge = True
                break
            if np.sum(np.array(best_point_now[0])>np.array(edgeL)) != DIMESSION:
                #continue
                #print('beyond the edge，please extend the edgeL')
                beyond_edge = True
                break
            if i % 300 == 0:
                total_criterion_now,best_criterion_now, best_point_now,mu_now,sigma_now,sigma2_now = sess.run([total_criterion,best_criterion,best_sample_point,mu,sigma,sigma2])

                if best_criterion_prev == None:
                    best_criterion_prev = best_criterion_now

                elif abs(best_criterion_now - best_criterion_prev) < criterion_eps:
                    if i >= min_iter:
                        break
                else:
                    best_criterion_prev = best_criterion_now
            best_criterion_now = check_edge(best_criterion_now,edgeL,edgeU)
        return best_point_now,gp_theta

