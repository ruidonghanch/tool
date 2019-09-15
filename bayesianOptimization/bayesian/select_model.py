import tensorflow as tf
import random
import numpy as np
'''
三种不同的采样函数
'''
# 均匀采样函数
def uniform_select(uniform_startL,uniform_startU):
    """
    均与分布采样
    Parameters
    ----------
    uniform_startL : np.array,尺寸为(1,DIMESSION),采样点最小值
    uniform_startU : np.array,尺寸为(1,DIMESSION),采样点最大值

    Returns
    -------
    select : np.array,尺寸为(1,DIMESSION)，选出的采样点
    """
    select = []
    for i, j in zip(uniform_startL, uniform_startU):
        select.append(random.uniform(i, j))
    return select


# 选取最高的n个点组成的边界进行采样
def rank_n_select(X,y,edgeL,edgeU,n=5):
    """
    在结果最好的n个点围城的区域中进行采样
    Parameters
    ----------
    X:np.array,尺寸为(n_sample,n_attribute),代表采样点
    y:np.array,尺寸为(n_sample,1)
    n: int,default = 5 选取最高的n个点
    edgeL:尺寸为(1,n_attribute),采样边界下界
    edgeU:尺寸为(1,n_attribute)，采样边界上界

    Returns
    -------
    np.array,尺寸为(1,DIMESSION)，选出的采样点
    """
    edgeL = np.array(edgeL)
    edgeU = np.array(edgeU)


    # 判断采样条件,无法进行n点最佳采样时，则采样全部值的边界采样
    if X.shape[0] < n:
        rank_n_select_startL = np.min(X, 0)
        rank_n_select_startU = np.max(X, 0)
    else:
        merge_data = np.hstack((X, y))
        select_data = merge_data[merge_data[:,X.shape[1]].argsort()][-n:, :-1]
        rank_n_select_startL = np.min(select_data, 0)
        rank_n_select_startU = np.max(select_data, 0)

    #判断采样空间是否超出边界
    rank_n_select_startL = check_edge(rank_n_select_startL, edgeL, edgeU)
    rank_n_select_startU = check_edge(rank_n_select_startU, edgeL, edgeU)
    return uniform_select(rank_n_select_startL, rank_n_select_startU)

# 极值采样函数
def max_select(X,y,edgeL,edgeU,scale = 0.1):
    """
    在结果最好的点旁边添加标准差为scale的扰动进行采样
    Parameters
    ----------
    X:np.array,尺寸为(n_sample,n_attribute),代表采样点
    y:np.array,尺寸为(n_sample,1)
    edgeL:array,尺寸为(1,n_attribute),采样上界
    degeU:array,尺寸为(1,n_attribute)，采样下界
    scale:float,default 0.1,代表max_select中抽样时添加高斯扰动项时，3*sigma距离在整个参数空间中的比例

    Returns
    -------
    np.array,尺寸为(1,DIMESSION)，选出的采样点
    """
    max_pos = np.argwhere(y == max(y))[0][0]
    candidate = np.array(X[max_pos,])
    candidate = check_edge(candidate,edgeL,edgeU)

    var_scale = np.array(edgeU)-np.array(edgeL)
    for i in range(X.shape[1]):
        sigma = var_scale[i]*scale/6 #按照3 sigma标准选出的扰动方差项
        candidate[i] = candidate[i]+np.random.normal(0, scale=sigma,size=1)
    candidate = check_edge(candidate,edgeL,edgeU)
    return candidate

def check_edge(vec,edgeL,edgeU):
    #检查是否越界，将vec强制转换至大于edgeL，小于edgeU范围内
    pos = 0
    for i,j in zip(vec,edgeL):
        if i<j:
            vec[pos] = edgeL[pos]
            pos += 1
    pos = 0
    for i, j in zip(vec, edgeU):
        if i>j:
            vec[pos] = edgeU[pos]
            pos += 1
    return vec