#!/usr/bin/env python3
#
# Evolutionary Algorithms

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_fitness(out_dir, name, algo_name, x, y1, y2, title):
    """
    (d) For each test function, plot the best and the worse fitness for each generation (averaged over 3 runs).
    :param name:
    :param x:
    :param y1:
    :param y2:
    :param title:
    """

    plt.figure()
    plt.grid()

    # Let x-axis be the generations and y-axis be the fitness values.
    plt.plot(x, y1, label='avg_' + name.lower() + '_max')
    plt.plot(x, y2, label='avg_' + name.lower() + '_min')

    plt.xlabel('generations', fontsize=11)
    plt.ylabel('fitness values', fontsize=11)

    plt.gca().set_ylim(bottom=-70)

    plt.annotate(round(y1[-1], 2), xy=(x[-1], y1[-1]), xycoords='data',
                xytext=(-40, 15), size=10, textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )

    plt.annotate(round(y2[-1], 2), xy=(x[-1], y2[-1]), xycoords='data',
                xytext=(-40, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )

    plt.legend()

    plt.title(algo_name + '\n' + title, weight='bold', fontsize=12)
    plt.savefig(out_dir + 'fitness.pdf')
    plt.close()


def plot_generation(out_dir, name, i, iteration, min, obj_fun, sample):
    """

    :param i:
    :param iteration:
    :param min:
    :param obj_fun:
    :param sample:
    :return:
    """

    if i % (iteration / 10) == 0:
        plt.figure(1)
        plt.clf()

        plot_2d_contour(obj_fun)
        plt.plot(sample[:, 0], sample[:, 1], 'ko')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title(name.upper() + '\ngeneration: ' + str(i + 1) + '\nmin: ' + str(min[i]))

        # plt.pause(0.1)
        plt.savefig(out_dir + name + '-generation-contour-' + str(i) + '.pdf')

    plt.close()


def cem(obj_fun, dim_domain, population_size, elite_set_ratio, learning_rate, iteration, out_dir, name, plot_generations):
    """

    :param dim_domain:
    :param population_size:
    :param elite_set_ratio:
    :param obj_fun:
    :param iter:
    :return mean:
    """
    # Initialise parameters
    # Note that you can uniformly sample the initial population parameters as long as they are reasonably far from
    # the global optimum.
    mean = np.random.uniform(-5, 5, dim_domain)
    variance = np.random.uniform(1, 2, dim_domain)

    max = np.zeros(iteration)
    min = np.zeros(iteration)

    for i in range(iteration):
        # Obtain n sample from a normal distribution
        sample = np.random.normal(mean, variance, [population_size, dim_domain])

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
        if plot_generations:
            plot_generation(out_dir, name, i, iteration, min, obj_fun, sample)

        # Refit a new Gaussian distribution from the elite set
        mean = np.mean(elite, axis=0)
        variance = np.std(elite, axis=0)

    # Return mean of final sampling distribution as solution
    return mean, min, max


def nes(obj_fun, dim_domain, population_size, elite_set_ratio, learning_rate, iteration, out_dir, name, plot_generations):
    """

    :param dim_domain:
    :param population_size:
    :param obj_fun:
    :param iter:
    :return mean:
    """
    # Initialise parameters
    mean = np.random.uniform(-5, 5, dim_domain)
    # variance = np.full(dim_domain, 1)
    variance = np.random.uniform(1, 2, dim_domain)

    max = np.zeros(iteration)
    min = np.zeros(iteration)

    for i in range(iteration):
        # Obtain n sample from a normal distribution
        sample = np.random.normal(mean, variance, [population_size, dim_domain])

        # Evaluate objective function on an objective function
        fitness = obj_fun(sample)

        min[i] = np.min(fitness)
        max[i] = np.max(fitness)

        # Calculate the log derivatives
        log_derivative_mu = (sample - mean) / (variance ** 2)
        log_derivative_sigma = ((sample - mean) ** 2 - (variance ** 2)) / variance ** 3

        J_gradient_mu = np.sum(fitness[..., np.newaxis] * log_derivative_mu, axis=0) / sample.shape[0]
        J_gradient_sigma = np.sum(fitness[..., np.newaxis] * log_derivative_sigma, axis=0) / sample.shape[0]

        F_mu = np.matmul(log_derivative_mu.T, log_derivative_mu) / sample.shape[0]
        F_sigma = np.matmul(log_derivative_sigma.T, log_derivative_sigma) / sample.shape[0]

        # PLOT
        if plot_generations:
            plot_generation(out_dir, name, i, iteration, min, obj_fun, sample)

        # Update mean and variance
        mean = mean - learning_rate * np.matmul(np.linalg.inv(F_mu), J_gradient_mu)
        variance = variance - learning_rate * np.matmul(np.linalg.inv(F_sigma), J_gradient_sigma)

    # Return mean of final sampling distribution as solution
    return mean, min, max


def cma_es(obj_fun, dim_domain, population_size, elite_set_ratio, learning_rate, iteration, out_dir, name, plot_generations):
    """

    :param dim_domain:
    :param population_size:
    :param elite_set_ratio:
    :param obj_fun:
    :param iter:
    :return mean:
    """
    # Initialise parameters
    # Note that you can uniformly sample the initial population parameters as long as they are reasonably far from
    # the global optimum.

    mean = np.random.uniform(-5, 5, dim_domain)
    cov_matrix = np.diag(np.random.uniform(1, 2, dim_domain))

    max = np.zeros(iteration)
    min = np.zeros(iteration)

    for i in range(iteration):
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
        if plot_generations:
            plot_generation(out_dir, name, i, iteration, min, obj_fun, sample)

        # Refit a new Gaussian distribution from the elite set

        # for i in range(dim_domain):
        #     for j in range(dim_domain):
        #         cov_matrix[i][j] = np.sum((elite[:, i] - mean[i]) * (elite[:, j] - mean[j]))

        diff = elite - mean
        cov_matrix = np.matmul(diff.T, diff) / elite.shape[0]

        mean = np.mean(elite, axis=0)

    # Return mean of final sampling distribution as solution
    return mean, min, max


def run(algorithm, experiment, run=3, dim_domain=100, population_size=30, elite_set_ratio=0.20,
        learning_rate=0.01, iteration=100):
    """

    :param algorithm:
    :param experiment:
    :param run:
    :param dim_domain:
    :param population_size:
    :param elite_set_ratio:
    :param learning_rate:
    :param iteration:
    :return:
    """
    print('Running '+ experiment + '…')
    out_dir_sphere = 'out/' + algorithm.__name__ + '/' + str(experiment) + '/sphere/'
    out_dir_rastrigin = 'out/' + algorithm.__name__ + '/' + str(experiment) + '/rastrigin/'
    check_dir(out_dir_sphere)
    check_dir(out_dir_rastrigin)

    avg_sphere_min = 0
    avg_rastrigin_min = 0

    avg_sphere_max = 0
    avg_rastrigin_max = 0

    avg_sphere_time = 0
    avg_rastrigin_time = 0

    for i in range(run):
        if i == 0:
            plot_generations = True
        else:
            plot_generations = False

        try:
            sphere_s_time = time.time()
            sphere_mean, sphere_min, sphere_max = algorithm(sphere_test, dim_domain, population_size, elite_set_ratio,
                                                            learning_rate, iteration, out_dir_sphere, algorithm.__name__, plot_generations)
            sphere_e_time = time.time()
            avg_sphere_time += sphere_e_time - sphere_s_time

            avg_sphere_min += sphere_min
            avg_sphere_max += sphere_max
        except ValueError:
            pass

        try:
            rastrigin_s_time = time.time()
            rastrigin_mean, rastrigin_min, rastrigin_max = algorithm(rastrigin_test, dim_domain, population_size,
                                                                     elite_set_ratio, learning_rate, iteration,
                                                                     out_dir_rastrigin, algorithm.__name__, plot_generations)
            rastrigin_e_time = time.time()
            avg_rastrigin_time += rastrigin_e_time - rastrigin_s_time

            avg_rastrigin_min += rastrigin_min
            avg_rastrigin_max += rastrigin_max
        except ValueError:
            pass

    avg_sphere_min /= run
    avg_rastrigin_min /= run
    avg_sphere_max /= run
    avg_rastrigin_max /= run

    avg_sphere_time /= run
    avg_rastrigin_time /= run

    iterator = np.arange(iteration)

    if not isinstance(avg_sphere_min, np.ndarray):
        best_fitness_sphere.append('Err')
        worse_fitness_sphere.append('Err')
        average_run_times_sphere.append('Err')
    else:
        plot_fitness(out_dir_sphere, 'Sphere', algorithm.__name__.upper(), iterator, avg_sphere_max, avg_sphere_min,
                     'Sphere Fitness')
        best_fitness_sphere.append(str(round(avg_sphere_min[-1], 2)))
        worse_fitness_sphere.append(str(round(avg_sphere_max[-1], 2)))
        average_run_times_sphere.append(str(round(avg_sphere_time, 2)) + ' sec')

    if not isinstance(avg_rastrigin_min, np.ndarray):
        best_fitness_rastrigin.append('Err')
        worse_fitness_rastrigin.append('Err')
        average_run_times_rastrigin.append('Err')
    else:
        plot_fitness(out_dir_rastrigin, 'Rastrigin', algorithm.__name__.upper(), iterator, avg_rastrigin_max,
                     avg_rastrigin_min, 'Rastrigin Fitness')
        best_fitness_rastrigin.append(str(round(avg_rastrigin_min[-1], 2)))
        worse_fitness_rastrigin.append(str(round(avg_rastrigin_max[-1], 2)))
        average_run_times_rastrigin.append(str(round(avg_rastrigin_time, 2)) + ' sec')

    experiments.append('texttt{' + experiment + '}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 1 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
data = np.dstack((X, Y))

# (a) Generate an 2D contour plot of 2-dimensional Sphere function
S = sphere_test(data)

plt.figure()
plt.contourf(X, Y, S, alpha=0.8)
plt.colorbar()
plt.savefig('out/' + sphere_test.__name__ + '.pdf')

# (b) Generate an 2D contour plot of 2-dimensional Rastrigin function
R = rastrigin_test(data)

plt.figure()
plt.contourf(X, Y, R, alpha=0.8)
plt.colorbar()
plt.savefig('out/' + rastrigin_test.__name__ + '.pdf')

# (c) For each test function, uniformly sample 100 points in the domain
n = 2
xy = np.random.uniform(-5, 5, [100, 2])

# Evaluate them with the test function and guess what might be the region of the global optimum.
# X, Y = np.meshgrid(xy[:, 0], xy[:, 0])
# data = np.dstack((X, Y))

S_eval = sphere_test(xy)

plt.figure()
plt.scatter(xy[:, 0], xy[:, 1], c=S_eval, alpha=0.8)
plt.colorbar()
plt.savefig('out/' + sphere_test.__name__ + '-eval.pdf')
plt.close()

R_eval = rastrigin_test(xy)

plt.figure()
plt.scatter(xy[:, 0], xy[:, 1], c=R_eval, alpha=0.8)
plt.colorbar()
plt.savefig('out/' + rastrigin_test.__name__ + '-eval.pdf')
plt.close()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 2 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# experiments = []
#
# best_fitness_sphere = []
# worse_fitness_sphere = []
#
# best_fitness_rastrigin = []
# worse_fitness_rastrigin = []
#
# average_run_times_sphere = []
# average_run_times_rastrigin = []
#
# # (a) Run CEM 3 times for both of test functions with 100-dimensional dim_domain.
# run(cem, experiment='baseline')
#
# # (b) Try different population size and elite set ratio and see what best performance you can obtain.
# run(cem, experiment='pop_size-100', population_size=100)
# run(cem, experiment='pop_size-1000', population_size=1000)
#
# run(cem, experiment='elite-30', elite_set_ratio=0.30)
# run(cem, experiment='elite-10', elite_set_ratio=0.10)
#
# run(cem, experiment='pop_size-100+elite-30', population_size=100, elite_set_ratio=0.30)
# run(cem, experiment='pop_size-100+elite-10', population_size=100, elite_set_ratio=0.10)
#
# run(cem, experiment='pop_size-1000+elite-30', population_size=1000, elite_set_ratio=0.30)
# run(cem, experiment='pop_size-1000+elite-10', population_size=1000, elite_set_ratio=0.10)
#
# # (c) Try different number of generations.
# run(cem, experiment='iteration-50', iteration=50)
# run(cem, experiment='iteration-30', iteration=30)
#
# run(cem, experiment='iteration-30+elite-10', elite_set_ratio=0.10, iteration=30)
# run(cem, experiment='iteration-50+elite-10', elite_set_ratio=0.10, iteration=50)
# run(cem, experiment='iteration-200+pop_size-1000+elite-30', population_size=1000, elite_set_ratio=0.30, iteration=200)
#
# sphere_data = {'textbf{experiment}': experiments,
#                'textbf{best fitness}': best_fitness_sphere,
#                'textbf{worse fitness}': worse_fitness_sphere,
#                'textbf{average run time}': average_run_times_sphere}
#
# rastrigin_data = {'textbf{experiment}': experiments,
#                   'textbf{best fitness}': best_fitness_rastrigin,
#                   'textbf{worse fitness}': worse_fitness_rastrigin,
#                   'textbf{average run time}': average_run_times_rastrigin}
#
# # Convert the dictionary into DataFrame
# sphere = pd.DataFrame(sphere_data)
# sphere = sphere.set_index('textbf{experiment}')
#
# check_dir('out/cem/')
# with open('out/cem/sphere-performance.tex', 'w') as latex_table:
#     latex_table.write(sphere.to_latex())
#
# rastrigin = pd.DataFrame(rastrigin_data)
# rastrigin = rastrigin.set_index('textbf{experiment}')
#
# check_dir('out/cem/')
# with open('out/cem/rastrigin-performance.tex', 'w') as latex_table:
#     latex_table.write(rastrigin.to_latex())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

experiments = []

best_fitness_sphere = []
worse_fitness_sphere = []

best_fitness_rastrigin = []
worse_fitness_rastrigin = []

average_run_times_sphere = []
average_run_times_rastrigin = []

# (a) Run NES 3 times for both of test functions with 100-dimensional dim_domain. (i.e. n = 100)
# Note that you can uniformly sample the initial population parameters as long as they are reasonably far from the
# global optimum.
run(nes, experiment='baseline')

# # (b) Try different population size and learning rate and see what best performance you can obtain.
run(nes, experiment='pop_size-100', population_size=100)
run(nes, experiment='pop_size-1000', population_size=1000)
run(nes, experiment='pop_size-5000', population_size=5000)

run(nes, experiment='lr-001', learning_rate=0.001)
run(nes, experiment='lr-0001', elite_set_ratio=0.0001)
run(nes, experiment='lr-00001', elite_set_ratio=0.00001)

run(nes, experiment='pop_size-1000+lr-001', population_size=1000, learning_rate=0.001)
run(nes, experiment='pop_size-1000+lr-0001', population_size=1000, learning_rate=0.0001)
run(nes, experiment='pop_size-5000+lr-001', population_size=5000, learning_rate=0.001)
run(nes, experiment='pop_size-5000+lr-0001', population_size=5000, learning_rate=0.0001)

# # (c) Try different number of generations.
run(nes, experiment='iteration-2000', iteration=2000)
run(nes, experiment='iteration-5000', iteration=5000)

run(nes, experiment='iteration-2000+pop_size-1000', population_size=1000, iteration=2000)
run(nes, experiment='iteration-5000+pop_size-1000', population_size=1000, iteration=5000)

run(nes, experiment='iteration-2000+pop_size-5000', population_size=5000, iteration=2000)
run(nes, experiment='iteration-5000+pop_size-5000', population_size=5000, iteration=5000)

run(nes, experiment='iteration-2000+pop_size-1000+lr-001', population_size=1000, learning_rate=0.001, iteration=2000)
run(nes, experiment='iteration-2000+pop_size-1000+lr-0001', population_size=1000, learning_rate=0.0001, iteration=2000)
run(nes, experiment='iteration-2000+pop_size-5000+lr-001', population_size=5000, learning_rate=0.001, iteration=2000)
run(nes, experiment='iteration-2000+pop_size-5000+lr-0001', population_size=5000, learning_rate=0.0001, iteration=2000)

sphere_data = {'textbf{experiment}': experiments,
               'textbf{best fitness}': best_fitness_sphere,
               'textbf{worse fitness}': worse_fitness_sphere,
               'textbf{average run time}': average_run_times_sphere}

rastrigin_data = {'textbf{experiment}': experiments,
                  'textbf{best fitness}': best_fitness_rastrigin,
                  'textbf{worse fitness}': worse_fitness_rastrigin,
                  'textbf{average run time}': average_run_times_rastrigin}

# Convert the dictionary into DataFrame
sphere = pd.DataFrame(sphere_data)
sphere = sphere.set_index('textbf{experiment}')

check_dir('out/nes/')
with open('out/nes/sphere-performance.tex', 'w') as latex_table:
    latex_table.write(sphere.to_latex())

rastrigin = pd.DataFrame(rastrigin_data)
rastrigin = rastrigin.set_index('textbf{experiment}')

check_dir('out/nes/')
with open('out/nes/rastrigin-performance.tex', 'w') as latex_table:
    latex_table.write(rastrigin.to_latex())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 4 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# experiments = []
#
# best_fitness_sphere = []
# worse_fitness_sphere = []
#
# best_fitness_rastrigin = []
# worse_fitness_rastrigin = []
#
# average_run_times_sphere = []
# average_run_times_rastrigin = []
#
# # (a) Run CMA-ES 3 times for both of test functions with 100-dimensional domain. (i.e. n = 100) Note that you can uniformly sample the initial population parameters as long as they are reasonably far from the global optimum.
# run(cma_es, experiment='baseline')
#
# # (b) Try different population size and learning rate and see what best performance you can obtain.
# run(cma_es, experiment='pop_size-100', population_size=100)
# run(cma_es, experiment='pop_size-1000', population_size=1000)
#
# run(cma_es, experiment='elite-30', elite_set_ratio=0.30)
# run(cma_es, experiment='elite-10', elite_set_ratio=0.10)
#
# run(cma_es, experiment='pop_size-100+elite-30', population_size=100, elite_set_ratio=0.30)
# run(cma_es, experiment='pop_size-100+elite-10', population_size=100, elite_set_ratio=0.10)
#
# run(cma_es, experiment='pop_size-1000+elite-30', population_size=1000, elite_set_ratio=0.30)
# run(cma_es, experiment='pop_size-1000+elite-10', population_size=1000, elite_set_ratio=0.10)
#
# # (c) Try different number of generations. What is the minimum number of gener- ations that you can obtain a solution close enough to the global optimum?
# run(cma_es, experiment='iteration-50', iteration=50)
# run(cma_es, experiment='iteration-30', iteration=30)
#
# run(cma_es, experiment='iteration-30+elite-10', elite_set_ratio=0.10, iteration=30)
# run(cma_es, experiment='iteration-50+elite-10', elite_set_ratio=0.10, iteration=50)
# run(cma_es, experiment='iteration-200+pop_size-1000+elite-30', population_size=1000, elite_set_ratio=0.30, iteration=200)
#
#
# sphere_data = {'textbf{experiment}': experiments,
#                'textbf{best fitness}': best_fitness_sphere,
#                'textbf{worse fitness}': worse_fitness_sphere,
#                'textbf{average run time}': average_run_times_sphere}
#
# rastrigin_data = {'textbf{experiment}': experiments,
#                   'textbf{best fitness}': best_fitness_rastrigin,
#                   'textbf{worse fitness}': worse_fitness_rastrigin,
#                   'textbf{average run time}': average_run_times_rastrigin}
#
# # Convert the dictionary into DataFrame
# sphere = pd.DataFrame(sphere_data)
# sphere = sphere.set_index('textbf{experiment}')
#
# check_dir('out/cma_es/')
# with open('out/cma_es/sphere-performance.tex', 'w') as latex_table:
#     latex_table.write(sphere.to_latex())
#
# rastrigin = pd.DataFrame(rastrigin_data)
# rastrigin = rastrigin.set_index('textbf{experiment}')
#
# check_dir('out/cma_es/')
# with open('out/cma_es/rastrigin-performance.tex', 'w') as latex_table:
#     latex_table.write(rastrigin.to_latex())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # TASK 5 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# (a) Plot the comparison of CEM, NES and CMA-ES for the best fitness in each generation.
# Note that this should be one single figure with three curves, each for one algorithm.

# (b) Plot the comparison of CEM, NES and CMA-ES for the worst fitness in each generation.
# Note that this should be one single figure with three curves, each for one algorithm.

# (c) For each test function, which algorithm is best? which is the worst?
