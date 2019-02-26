# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import copy

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    gradU = reg * Ui - (Yij - np.sum(Ui * Vj)) * Vj
    # if np.isnan(gradU).any():
    #     print(Ui, Yij, Vj, reg, eta)
    return eta * gradU


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    gradV = reg * Vj - (Yij - np.sum(Ui * Vj)) * Ui
    # if np.isnan(gradV).any():
    #     print(Vj, Yij, Ui, reg, eta)
    return eta * gradV


def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    predicts = U @ V.T
    err = 0
    for dPoint in Y:
        dPoint = dPoint - np.array([1, 1, 0])
        err = err + (dPoint[2] - predicts[dPoint[0], dPoint[1]]) **2 
    return err / len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.000001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.rand(M, K) - 1
    V = np.random.rand(N, K) - 1
    
    iniLoss = get_err(U, V, Y, reg)

    loss = iniLoss

    for i in range(max_epochs):
        Ym = np.random.permutation(Y) 
        for dPoint in Ym:
            Ui = U[dPoint[0] - 1, :]
            Vj = V[dPoint[1] - 1, :]
            Yij = dPoint[2]

            dU = grad_U(Ui, Yij, Vj, reg, eta)
            dV = grad_V(Vj, Yij, Ui, reg, eta)

            updateU = Ui - dU
            updateV = Vj - dV

            U[dPoint[0] - 1, :] = updateU
            V[dPoint[1] - 1, :] = updateV

        oldLoss = loss
        loss = get_err(U, V, Y, reg)
        if i == 0:
            d0 = iniLoss - loss

        if oldLoss - loss < eps * (d0):
            print("early stop at {0}".format(i))
            break

        if i == max_epochs - 1:
            print('Normal stop at {0}'.format(max_epochs))
    
    return U, V, loss
