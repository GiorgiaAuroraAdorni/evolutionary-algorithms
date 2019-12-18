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


def rastrigin_test(data, A=10):
    n = data.shape[1]

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

# (a) Generate an 2D contour plot of 2-dimensional Sphere function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
data = np.dstack((X, Y))

S = sphere_test(data)

plt.contour(X, Y, S)
# plt.plot(X, Y, 'ko', ms=3)
plt.show()

# (b) Generate an 2D contour plot of 2-dimensional Rastrigin function
R = rastrigin_test(data)

plt.contour(X, Y, R)
# plt.plot(X, Y, 'ko', ms=3)
plt.show()

# (c) For each test function, uniformly sample 100 points in the domain
xy = np.random.uniform(-5, 5, [100, n])

# TODO evaluate them with the test function and guess what might be the region of the global optimum.
S_eval = sphere_test(xy)

R_eval = rastrigin_test(xy)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

domain = 100
population_size = 30
elite_set_ratio = 0.20

# (a) Run CEM 3 times for both of test functions with 100-dimensional domain.
run = 3

for i in range(run):
    sphere_mean, sphere_min, sphere_max = cem(domain, population_size, elite_set_ratio, sphere_test)
    rastrigin_mean, rastrigin_min, rastrigin_max = cem(domain, population_size, elite_set_ratio, rastrigin_test)

    print(sphere_mean)
    print(rastrigin_mean)

# (b) Try different population size and elite set ratio and see what best performance you can obtain.
population_size = 50
elite_set_ratio = 0.20

for i in range(run):
    sphere_mean, sphere_min, sphere_max = cem(domain, population_size, elite_set_ratio, sphere_test)
    rastrigin_mean, rastrigin_min, rastrigin_max = cem(domain, population_size, elite_set_ratio, rastrigin_test)

    print(sphere_mean)
    print(rastrigin_mean)

population_size = 30
elite_set_ratio = 0.30

for i in range(run):
    sphere_mean, sphere_min, sphere_max = cem(domain, population_size, elite_set_ratio, sphere_test)
    rastrigin_mean, rastrigin_min, rastrigin_max = cem(domain, population_size, elite_set_ratio, rastrigin_test)

    print(sphere_mean)
    print(rastrigin_mean)

# (c) Try different number of generations.
population_size = 30
elite_set_ratio = 0.20

for i in range(run):
    sphere_mean, sphere_min, sphere_max = cem(domain, population_size, elite_set_ratio, sphere_test, iter=50)
    rastrigin_mean, rastrigin_min, rastrigin_max = cem(domain, population_size, elite_set_ratio, rastrigin_test, iter=50)

    print(sphere_mean)
    print(rastrigin_mean)

# What is the minimum number of generations that you can obtain a solution close enough to the global optimum?

# (d) For each test function, plot the best and the worse fitness for each generation (averaged over 3 runs).
avg_sphere_min = 0
avg_rastrigin_min = 0

avg_sphere_max = 0
avg_rastrigin_max = 0
iter = 30
iterator = np.arange(iter)

for i in range(run):
    sphere_mean, sphere_min, sphere_max = cem(domain, population_size, elite_set_ratio, sphere_test, iter)
    rastrigin_mean, rastrigin_min, rastrigin_max = cem(domain, population_size, elite_set_ratio, rastrigin_test, iter)

    avg_sphere_min += sphere_min
    avg_rastrigin_min += rastrigin_min

    avg_sphere_max += sphere_max
    avg_rastrigin_max += rastrigin_max

    print(sphere_mean)
    print(rastrigin_mean)

avg_sphere_min /= run
avg_rastrigin_min /= run

avg_sphere_max /= run
avg_rastrigin_max /= run


plt.xlabel('generations', fontsize=11)
plt.ylabel('fitness values', fontsize=11)

plt.plot(iterator, avg_sphere_max, label='avg_sphere_max')
plt.plot(iterator, avg_sphere_min, label='avg_sphere_min')
plt.legend()
plt.title('Sphere Fitness', weight='bold', fontsize=12)
# plt.savefig(out_dir + 'scores-comparison-' + model1 + '-' + model2 + '.pdf')
plt.show()
plt.close()


plt.xlabel('generations', fontsize=11)
plt.ylabel('fitness values', fontsize=11)

plt.plot(iterator, avg_rastrigin_max, label='avg_rastrigin_max')
plt.plot(iterator, avg_rastrigin_min, label='avg_rastrigin_min')
plt.legend()
plt.title('Rastrigin Fitness', weight='bold', fontsize=12)
# plt.savefig(out_dir + 'scores-comparison-' + model1 + '-' + model2 + '.pdf')
plt.show()
plt.close()


# Let x-axis be the generations and y-axis be the fitness values.
# Note that this should be one single figure with two curves (one for best fitness and another for worst fitness).
