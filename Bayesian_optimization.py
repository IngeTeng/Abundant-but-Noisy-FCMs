import numpy as np
import matplotlib.pyplot as plt
import membership_function as mf
import random_number as rn


def ganma_r_step(N, E, K):

    # Step 0: establish variables
    n = E.shape[1]
    D = E.shape[0]

    ## gamma and rho estimates.
    anum = np.zeros(shape=(D))
    aden = np.zeros(shape=(D))
    rnum = np.zeros(shape=(D))

    # Step 1: loop through the upper triangle of matrix K to calculate the sums
    for j in range(0, n):
        for i in range(0, n):
            for d in range(0, D):
                if d == 0:
                    anum[0] += E[0, i, j] * K[0, i, j]
                else:
                    anum[d] += (N[i, j] - E[d, i, j]) * K[d, i, j]

                rnum[d] += K[d, i ,j]
                aden[d] += N[i, j] * K[d, i, j]

    # Step 2: calculate gamma and rho
    gamma = np.zeros(shape=(D))
    rho = np.zeros(shape=(D))
    for d in range(0, D):

        gamma[d] = anum[d] * 1. / (aden[d])
        rho[d] = 1. / (n * n) * rnum[d]

    # Step 3: return gamma and rho
    return (gamma, rho)


def k_step(N, E, gamma, rho):

    # Step 0: establish variables
    n = E.shape[1]  # the number of nodes in the network, following Newman's notation

    # Create an array to store k estimates
    K = np.zeros(E.shape)
    D = E.shape[0]
    # Step 1: loop through the upper triangle of the network to calculate new K
    for j in range(0, n):
        for i in range(0, n):
            p = np.zeros(shape=(D))
            sum = 0
            for d in range(0, D):
               if d == 0:
                   p[d] = (rho[d] * (gamma[d] ** E[d, i, j])) * ((1 - gamma[d]) ** (N[i, j] - E[d, i, j]))
               else:
                   p[d] = (rho[d] * (gamma[d] ** (N[i, j] - E[d, i, j]))) * ((1 - gamma[d]) ** (E[d, i, j]))

               sum = sum + p[d]
            #计算 K
            for d in range(0, D):
                K[d, i, j] = p[d] * 1. / (sum * 1.)

    # Step 2: return K
    return K

#stop condition
def condition(gamma_prev, gamma, rho_prev, rho, tolerance):
    res_gamma = abs(gamma_prev - gamma)
    res_rho = abs(rho_prev - rho)
    arr = res_gamma + res_rho
    size = arr.shape[0]
    tag = 0
    for i in range(size):
        if arr[0] > tolerance:
            tag = tag + 1
    if tag > 0:
        return True #Continue
    else:
        return False #stop

def bayesian_optimization(N, E, tolerance=.000001, seed=10):
    #Expectation-Maximization (EM) algorithm

    D = E.shape[0]
    iterations = 0
    gamma_prev = np.zeros(shape=(D))
    rho_prev = np.zeros(shape=(D))

    # Step 1: Do an initial q-step with random alpha, beta, and rho values
    # Randomly assign values to alpha, beta, and rho to start
    np.random.seed(seed)

    gamma_tmp = np.sort(np.random.rand(D))
    gamma = gamma_tmp[::-1]
    rho = rn.gen_random_number(0, 1, D)

    # Now calculate initial K
    K = k_step(N, E, gamma, rho)

    # Step 2: Repeat until tolerance is met
    while condition(gamma_prev, gamma, rho_prev, rho, tolerance):
        gamma_prev = gamma
        rho_prev = rho

        gamma, rho  = ganma_r_step(N, E, K)
        K = k_step(N, E, gamma, rho)
        iterations += 1

    # Step 3: return values
    return (gamma, rho, K, iterations)

