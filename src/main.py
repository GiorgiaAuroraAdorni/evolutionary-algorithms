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


def cem(domain, population_size, elite_set_ratio, obj_fun, iter = 100):
    """

    :param d:
    :param N:
    :param obj_fun:
    :param p:
    :return mean:
    """
    # Initialise parameters
    # Note that you can uniformly sample the initial population parameters as long as they are reasonably far from
    # the global optimum.
    mean = np.random.uniform(-5, 5, domain)
    variance = np.random.uniform(0, 5, domain)

    max = np.zeros(iter)
    min = np.zeros(iter)

    for i in range(iter):
        # Obtain n sample from a normal distribution
        sample = np.random.normal(mean, variance, [population_size, domain])

        # Evaluate objective function on an objective function
        fitness = obj_fun(sample)

        min[i] = np.min(fitness)
        max[i] = np.max(fitness)

        # Sort sample by objective function values in descending order
        idx = np.argsort(fitness)
        fittest = sample[idx]

        # Elite set
        p = np.rint(population_size * elite_set_ratio).astype(np.int)
        elite = fittest[:p]

        # Refit a new Gaussian distribution from the elite set
        mean = np.mean(elite, axis=0)
        variance = np.std(elite, axis=0)

    # Return mean of final sampling distribution as solution
    return mean, min, max


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

