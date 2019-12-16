#!/usr/bin/env python3
#
# Evolutionary Algorithms

import os

import numpy as np
import matplotlib.pyplot as plt


def check_dir(directory):
    """
    :param directory: path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def sphere_test(data):
    f_x = np.sum(np.square(data), axis=-1)
    return f_x


def rastrigin_test(A, n, data):
    d1 = np.square(data)
    arg_cos = 2 * np.pi * data
    cos = np.cos(arg_cos)
    d2 = np.multiply(A, cos)

    a1 = A * n
    a2 = np.sum(d1 - d2, axis=-1)
    return np.sum([a1, a2])


def cem(gaussian_distrib, mean, cov):
    # Initialize parameters
    mu = -6
    sigma2 = 100
    t = 0
    maxits = 100
    N = 100
    Ne = 10
    # While maxits not exceeded and not converged
    while (t < maxits) and (sigma2 > epsilon):
        # Obtain N samples from current sampling distribution
        X = SampleGaussian(mu, sigma2, N)
        # Evaluate objective function at sampled points
        S = exp(-(X - 2) ^ 2) + 0.8 * np.exp(-(X + 2) ^ 2)
        # Sort X by objective function values in descending order
        X = sort(X, S)
        # Update parameters of sampling distribution
        mu = mean(X(1:Ne))
        sigma2 = var(X(1:Ne))
        t = t + 1
    # Return mean of final sampling distribution as solution
    return mu



################################################################################
n = 2
A = 10

x = np.random.uniform(-5, 5, 100)
y = np.random.uniform(-5, 5, 100)

x_ = sorted(x)
y_ = sorted(y)

X, Y = np.meshgrid(x_, y_)
data = np.dstack((X, Y))

# (a) Generate an 2D contour plot of 2-dimensional Sphere function
S = sphere_test(data)

plt.contour(data[:, :, 0], data[:, :, 1], S)
plt.plot(x, y, 'ko', ms=3)
plt.show()

# (b) Generate an 2D contour plot of 2-dimensional Rastrigin function
R = rastrigin_test(A, n, data)

plt.contour(data[:, :, 0], data[:, :, 1], R)
plt.plot(x, y, 'ko', ms=3)
plt.show()

# (c) For each test function, uniformly sample 100 points in the domain, evaluate them with the test function and
# guess what might be the region of the global optimum.

