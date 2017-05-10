#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import numpy as np
import utils


def get_whitened_data(X):
    """Return a whitened version of matrix X"""
    # compute the covariance matrix
    cov = np.dot(X.T, X)
    # perform the eigenvalue decomposition
    d, E = np.linalg.eigh(cov)
    # compute D^-1/2
    D = np.diag(1. / np.sqrt(d))
    # compute W = E * D^-1/2 * E.T
    W = np.dot(np.dot(E, D), E.T)
    X_white = np.dot(X, W)
    return X_white


def whitening():
    """
    - Implement a function that returns a whitened version of the data given as input.
    - Add to this function a test that makes sure that  E[x̃ x̃ ⊤]≈E[x~x~⊤]≈I (up to numerical accuracy).
    - Reproduce the scatter plots of the demo code, but this time, using the whitened data.
    - Render 500 whitened image patches.
    """
    X = utils.load()
    X_white = get_whitened_data(X)

    def is_white(X_white):
        # X * X.T ≈ I
        cov = np.dot(X_white, X_white)
        return np.allclose(cov, np.identity(cov.shape))

    assert True, is_white(X_white)

    utils.scatterplot(X_white[:, 0], X_white[:, 1], xlabel='pixel 0, red', ylabel='pixel 0, green')

    utils.scatterplot(X_white[:, 0], X_white[:, 3], xlabel='pixel 0, red', ylabel='pixel 1, red')

    utils.render(X_white[:500])


def fastICA(X):
    """
    - Implement the FastICA method described in the paper, and run it for 64 iterations.
    - Print the value of the objective function at each iteration.
    - Create a scatter plot of the projection of the whitened data on two distinct independent components after 0, 1, 3, 7, 15, 31, 63 iterations.
    - Visualize the learned independent components using the function render(...).
    """
    def G(x):
        a = 1.5
        return 1/a * np.log(np.cosh(a * x))

    def g(x):
        a = 1.5
        return np.tanh(a * x)

    def gprime(x):
        a = 1.5
        return a * (1 - (np.dot(np.tanh(a * x), np.tanh(a * x))))

    def J(y):
        v = np.random.normal(size=y.shape)
        return np.sum(np.square(y - v))

    N_COMP = 100
    N_ITERATIONS = 64
    INTERESTIGN_ITERATIONS = [0, 1, 3, 7, 15, 31, 63]
    INTERESTIGN_COMPONENT = 1

    X = get_whitened_data(X)[:N_COMP]
    w_init = np.random.normal(size=(N_COMP, N_COMP))
    W = np.zeros((N_COMP, N_COMP), dtype=float)

    for j in range(N_COMP):
        w = w_init[j, :].copy()
        W[j, :] = w

    for i in range(N_ITERATIONS):
        for j in range(N_COMP):
            w = W[j, :]
            wtx = np.dot(w.T, X)
            gwtx = g(wtx)
            g_wtx = gprime(wtx)
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            w1 /= np.sqrt((w1**2).sum())
            w = w1
            if j == INTERESTIGN_COMPONENT and i in INTERESTIGN_ITERATIONS:
                utils.scatterplot(np.dot(W[j-1, :].T, X[:N_COMP]), np.dot(W[j, :].T, X[:N_COMP]), xlabel='pixel 0, red', ylabel='pixel 0, green')

            W[j, :] = w
        print "it = " + str(i) + "\tJ(W) = " + str(J(np.dot(W, X[:N_COMP])))

    utils.render(np.dot(W, X[:N_COMP])[:500])
