#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import numpy as np
import math
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


def fastICA():
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

    def g_(x):
        a = 1.5
        return a * (1 - math.pow(np.tanh(a * x), 2))

    def J(y):
        v = np.random.normal()
        return math.pow(G(y) - G(v), 2)  # TODO: implement E{}

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm

    X = utils.load()
    X = get_whitened_data(X)
    n = 64
    components = 100
    N, M = X.shape
    ws = []
    g_vec = np.vectorize(g)
    g__vec = np.vectorize(g_)
    vec_1 = np.array([1 for x in range(M)])
    vec_1.shape = (M, 1)

    for p in range(components):
        w_p = np.random.rand(N)
        ws.append(w_p)
        for _ in range(n):
            w_p = np.multiply(1/float(M), np.dot(X, g_vec(np.dot(w_p.T, X)).T))
            w_p -= np.multiply(1/float(M), np.multiply(np.dot(g__vec(np.dot(w_p.T, X)), vec_1), w_p))
            sum_ = 0
            for j in range(p):
                sum_ += np.multiply(np.dot(w_p.T, ws[j]), ws[j])
            w_p -= sum_
            w_p = normalize(w_p)
            print np.sum(w_p)
        return


if __name__ == '__main__':
    fastICA()
