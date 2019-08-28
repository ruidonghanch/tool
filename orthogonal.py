import numpy as np
from numpy import linalg as la


def normalize(X):
    m = X.shape[0]
    for i in range(m):
        X[i, :] = X[i, :] / np.sqrt(np.sum(X[i, :] ** 2))
    return X

def orthogonalRotate(Y, L1, L2, alpha, normalization = False):
    '''

    :param Y: array [m,n], label matrix,row[i]=[0,0,...,0,1,0,...,0]
    :param L1: array [m,n], CE output matrix
    :param L2: array [m,n], TL output matrix
    :param alpha: float, weight of L2 vector
    :param normalization: bool, normalize input data
    :return: array [n,n], best orthogonal matrix T

    Tips : suggest normalize the input vector
    '''
    L1 = np.array(L1)
    L2 = np.array(L2)
    Y = np.array(Y)

    if normalization == True:
        L1 = normalize(L1)
        L2 = normalize(L2)

    m = Y.shape[0]
    for i in range(m):
        target = np.sqrt(np.sum(L1[i,:]**2)) + alpha*np.sqrt(np.sum(L2[i,:]**2))
        Y[i,:] = Y[i,:]*target

    Y_target = Y - L1
    X_target = L2*alpha
    A = X_target.T.dot(Y_target)
    U,sigma,VT=la.svd(A)
    return U.dot(VT)




if __name__ == "__main__":
    Y = np.array([[0,1,0],[1,0,0],[0,0,1]],dtype='int16')
    L1 = np.array([[0.1,0.6,0.3],[0.2,0.4,0.2],[0.2,0.1,0.8]],dtype='float32')
    L2 = np.array([[0.4,0.5,0.2],[0.7,0.1,0.2],[0.4,0.2,0.3]],dtype='float32')
    alpha = 0.5
    rotate_matrix = orthogonalRotate(Y, L1, L2, alpha, True)
    m = Y.shape[0]
    for i in range(m):
        target = np.sqrt(np.sum(L1[i, :] ** 2)) + alpha * np.sqrt(np.sum(L2[i, :] ** 2))
        Y[i, :] = Y[i, :] * target
    loss_1 = np.sum(((L1 + alpha*L2) - Y)**2)
    # print('origin predict:\n',L1 + alpha*L2)
    print('origin loss',loss_1)
    loss_2 = np.sum(((L1 + alpha*L2.dot(rotate_matrix)) - Y)**2)
    # print('rotate predict:\n',L1 + alpha*L2.dot(rotate_matrix))
    print('rotate loss',loss_2)