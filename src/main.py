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
    os.makedirs(directory, exist_ok=True)


def sphere_test(data):
    """

    :param data:
    :return:
    """
    f_x = np.sum(np.square(data), axis=-1)
    return f_x


def rastrigin_test(data, A=10):
    """

    :param data:
    :param A:
    :return:
    """
    n = data.shape[1]

    cos = np.cos(2 * np.pi * data)
    e1 = np.square(data) - np.multiply(A, cos)

    e2 = np.sum(e1, axis=-1)
    return np.sum([A * n, e2])


def plot_2d_contour(obj_function):
    """

    :param obj_function:
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    X, Y = np.meshgrid(x, y)
    data = np.dstack((X, Y))

    S = obj_function(data)
    plt.contour(X, Y, S)
    # plt.plot(X, Y, 'ko', ms=3)


def plot_fitness(experiment, name, x, y1, y2, title):
    """
    (d) For each test function, plot the best and the worse fitness for each generation (averaged over 3 runs).
    :param name:
    :param x:
    :param y1:
    :param y2:
    :param title:
    """
    out_dir = 'out/img/' + str(experiment) + '/'
    check_dir(out_dir)

    plt.xlabel('generations', fontsize=11)
    plt.ylabel('fitness values', fontsize=11)

    # Let x-axis be the generations and y-axis be the fitness values.
    plt.plot(x, y1, label='avg_' + name.lower() + '_max')
    plt.plot(x, y2, label='avg_' + name.lower() + '_min')

    plt.legend()

    plt.title(title, weight='bold', fontsize=12)
    plt.savefig(out_dir + name + '-fitness.pdf')

    plt.show()
    plt.close()


def cem(obj_fun, domain, population_size, elite_set_ratio, iter):
    """

    :param domain:
    :param population_size:
    :param elite_set_ratio:
    :param obj_fun:
    :param iter:
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

        # PLOT
        plt.figure(1)
        plt.clf()

        plot_2d_contour(obj_fun)
        plt.plot(sample[:, 0], sample[:, 1], 'ko')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title('generation ' + str(i))
        plt.pause(0.1)

        # Refit a new Gaussian distribution from the elite set
        mean = np.mean(elite, axis=0)
        variance = np.std(elite, axis=0)

    # Return mean of final sampling distribution as solution
    return mean, min, max


def run_cem(experiment, run=3, domain=100, population_size=30, elite_set_ratio=0.20, iter=100):
    """

    :param run:
    :param domain:
    :param population_size:
    :param elite_set_ratio:
    :param iter:
    :return:
    """
    avg_sphere_min = 0
    avg_rastrigin_min = 0

    avg_sphere_max = 0
    avg_rastrigin_max = 0

    for i in range(run):
        sphere_mean, sphere_min, sphere_max = cem(sphere_test, domain, population_size, elite_set_ratio, iter)
        rastrigin_mean, rastrigin_min, rastrigin_max = cem(sphere_test, domain, population_size, elite_set_ratio, iter)

        avg_sphere_min += sphere_min
        avg_rastrigin_min += rastrigin_min

        avg_sphere_max += sphere_max
        avg_rastrigin_max += rastrigin_max

    avg_sphere_min /= run
    avg_rastrigin_min /= run
    avg_sphere_max /= run
    avg_rastrigin_max /= run

    iterator = np.arange(iter)

    plot_fitness(experiment, 'Sphere', iterator, avg_sphere_max, avg_sphere_min, 'Sphere Fitness')
    plot_fitness(experiment, 'Rastrigin', iterator, avg_rastrigin_max, avg_rastrigin_min, 'Rastrigin Fitness')


def nes(obj_fun, domain, population_size, elite_set_ratio, iter):
    """

    :param domain:
    :param population_size:
    :param elite_set_ratio:
    :param obj_fun:
    :param iter:
    :return mean:
    """
    # TODO
    #  Initialise parameters
    #  Note that you can uniformly sample the initial population parameters as long as they are reasonably far from
    #  the global optimum.
    mean = np.random.uniform(-5, 5, domain)
    variance = np.random.uniform(0, 5, domain)

    max = np.zeros(iter)
    min = np.zeros(iter)

    for i in range(iter):
        # TODO Obtain n sample from a normal distribution
        sample = np.random.multivariate_normal(mean, variance, [population_size, domain])

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


def cma_es(obj_fun, domain, population_size, elite_set_ratio, iter):
    """

    :param domain:
    :param population_size:
    :param elite_set_ratio:
    :param obj_fun:
    :param iter:
    :return mean:
    """
    # Initialise parameters
    # Note that you can uniformly sample the initial population parameters as long as they are reasonably far from
    # the global optimum.

    mean = np.random.uniform(-5, 5, domain)

    cov_matrix = np.random.uniform(0, 5, [domain, domain])
    # cov_matrix = np.diag(np.random.uniform(0, 5, domain))
    cov_matrix = np.cov(cov_matrix)

    max = np.zeros(iter)
    min = np.zeros(iter)

    for i in range(iter):
        # Obtain n sample from a normal multivariate distribution
        sample = np.random.multivariate_normal(mean, cov_matrix, population_size)

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

        # PLOT
        plt.figure(1)
        plt.clf()

        plot_2d_contour(obj_fun)
        plt.plot(sample[:, 0], sample[:, 1], 'ko')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title('generation ' + str(i))
        plt.pause(0.1)

        # TODO Refit a new Gaussian distribution from the elite set
        #  save mean
        # TODO Using only the best solutions, along with the mean of the current generation (the green dot),
        #  calculate the covariance matrix of the next generation.
        #  Sample a new set of candidate solutions using the updated mean and covariance matrix

        # for i in range(domain):
        #     for j in range(domain):
        #         cov_matrix[i][j] = np.sum((elite[:, i] - mean[i]) * (elite[:, j] - mean[j]))

        diff = elite - mean
        cov_matrix = np.matmul(diff.T, diff) / elite.shape[0]

        mean = np.mean(elite, axis=0)

    # Return mean of final sampling distribution as solution
    return mean, min, max


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 1 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # (a) Generate an 2D contour plot of 2-dimensional Sphere function
# plot_2d_contour(sphere_test)
#
# # (b) Generate an 2D contour plot of 2-dimensional Rastrigin function
# plot_2d_contour(rastrigin_test)
#
#
# # TODO (c) For each test function, uniformly sample 100 points in the domain
# n = 2
# xy = np.random.uniform(-5, 5, [100, n])
#
# # TODO evaluate them with the test function and guess what might be the region of the global optimum.
# S_eval = sphere_test(xy)
#
# R_eval = rastrigin_test(xy)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 2 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # (a) Run CEM 3 times for both of test functions with 100-dimensional domain.
# run_cem(experiment=1)
#
# # (b) Try different population size and elite set ratio and see what best performance you can obtain.
# run_cem(experiment=2, population_size=50)
# run_cem(experiment=3, elite_set_ratio=0.30)
#
# # (c) Try different number of generations.
# run_cem(experiment=4, iter=50)
# run_cem(experiment=5, iter=30)

# cma_es(sphere_test, 100, 30, 0.20, 100)

# TODO What is the minimum number of generations that you can obtain a solution close enough to the global optimum?

# # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# (a) Run NES 3 times for both of test functions with 100-dimensional domain. (i.e. n = 100) Note that you can uniformly sample the initial population parameters as long as they are reasonably far from the global optimum.
# (b) Try different population size and learning rate and see what best performance you can obtain.
# (c) Try different number of generations. What is the minimum number of gener- ations that you can obtain a solution close enough to the global optimum?
# (d) For each test function, plot the best and the worse fitness for each generation (averaged over 3 runs). Let x-axis be the generations and y-axis be the fitness values. Note that this should be one single figure with two curves (one for best fitness and another for worst fitness).



cem(rastrigin_test, 100, 100, 0.2, 100)

plt.show()
plt.close()
