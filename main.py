import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
# Let's set our seed

np.random.seed(777)

# Simulons un mouvement brownien - source : https://towardsdatascience.com/animated-visualization-of-brownian-motion-in-python-3518ecf28533

def brownian_motion(N=50, T=1, h=1):
    dt = 1. * T/N  # the normalizing constant
    random_increments = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)  # the epsilon values
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    brownian_motion = np.insert(brownian_motion, 0, 0.0) # insert the initial condition
    
    return brownian_motion

# Fixons nos constantes

alpha = 0.2
b = 20
sigma = 0.3
T = 1
r = 0.05
K = 5
k = 20
S_0 = 20

defaults = {
    'alpha' : alpha,
    'b' : b,
    'sigma' : sigma,
    'T' : T,
    'r' : r,
    'K' : K,
    'k' : k,
    'S_0' : S_0
}


# Simulons le modèle CIR : https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0
# St+1 - St = delta * alpha * (b - St) + sigma * sqrt(St) * epsilont 

def cir(S_0, alpha, b, sigma, k, T, r, K):
    dt = T/float(k)
    spots = [S_0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * np.random.normal()
        spots.append(spots[-1] + ds)
    return spots, r, T, k, K

def anti_cir(S_0, alpha, b, sigma, k, T, r, K):
    dt = T/float(k)
    spots = [S_0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * - np.random.normal()
        spots.append(spots[-1] + ds)
    return spots, r, T, k, K

# Fonctions de simulation de C

def phi(S):
    S, r, T, k, K = S
    return math.exp(-r * T) * max(1/k * sum(S) - K, 0)

mc_C = lambda n, params : 1/n * sum([phi(cir(**params)) for _ in range(n)])

def mc_C_anti(n, params):
    su = [phi(cir(**params)) + phi(anti_cir(**params)) for _ in range(n)]
    return 1/(2 * n) * sum(su)

def mc_C_control(n, params):
    Z = np.random.normal(size=n)
    phi_X = [phi(cir(**params)) for _ in range(n)]
    beta = np.cov(phi_X, Z)[0][1]

    return 1/n * sum([phi_X[_] - beta * Z[_] for _ in range(n)])

# Simulations 

n = 100

start_MC = time.time()
C = mc_C(n, defaults) # Simulation classique de C
print('MC : {}'.format(time.time() - start_MC))

start_MC_anti = time.time()
C_anti = mc_C_anti(n, defaults) # Réduction de variance par variable antithétique
print('MC_anti : {}'.format(time.time() - start_MC_anti))

start_MC_cont = time.time()
C_control = mc_C_control(n, defaults) # Réduction de variance par variable de controle
print('MC_cont : {}'.format(time.time() - start_MC_cont))

# Comparons les erreures de Monte-Carlo

def plot_mcerr(methods, params, N, n, labels):
    simus = [[method(n, params) for _ in range(N)] for method in methods]
    plt.boxplot(simus, labels=labels)
    plt.show()

#plot_mcerr([mc_C, mc_C_anti, mc_C_control], defaults, 100, 100, labels=['Monte Carlo', 'Antithetic Variates', 'Control Variates'])


# Multi-level Monte-Carlo

M = 2
e = math.e ** -3
L = int(math.log(e ** -1) / math.log(M))
N_0 = 100

defaults_mlmc = {**defaults, 'M' : M}
brownian = lambda l : [np.random.normal() for _ in range(M ** l)]

def cir_mlmc(l, brownian, S_0, alpha, b, sigma, M, T, r, k, K):
    dt = M ** -l * T
    spots = [S_0]
    for i in range(M ** l):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * brownian[i]
        spots.append(spots[-1] + ds)
    return spots, r, T, k, K


H = lambda l : M ** (-l) * T
V = lambda l : np.var(cir_mlmc(l, brownian(l), **defaults_mlmc)[0])
P = lambda l, brownian : phi(cir_mlmc(l, brownian, **defaults_mlmc))
N = lambda l : int(2 * e ** (-2) * math.sqrt(V(l) * H(l)) * sum([math.sqrt(V(l)/H(l)) for l in range(L)])) + 1


def brownian_bis(brownian):
    return [sum(brownian[i:i+M]) for i in range(0, len(brownian), M)]

def Y(l):
    su = 0
    print('NL : ' + str(N(l)))
    for i in range(N(l)):
        bro = brownian(l)
        bbro = brownian_bis(bro)
        p = P(l, bro) - P(l-1, bbro)
        su += p
    return 1/N(l) * su

start_MLMC = time.time()
Y_0 = 1/N_0 * sum(P(0, brownian(0)) for _ in range(N_0))
Y = Y_0 + sum(Y(l) for l in range(1, L))
print('MLMC : {}'.format(time.time() - start_MLMC))


# TODO Illustrer comment mlmc peut réduire le temps de calcul

# Implémentation quasi MC

# Génération séquence halton : https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae

def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return norm.ppf(sample)

# Calculons option asiatique  


def cir_qmc(halt, S_0, alpha, b, sigma, k, T, r, K):
    dt = T/float(k)
    spots = [S_0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * halt[i]
        spots.append(spots[-1] + ds)
    return spots, r, T, k, K

mc_qmc_C = lambda n, halton_: 1/n * sum([phi(cir_qmc(halton_[i], **defaults)) for i in range(n)])

start_MC_qmc = time.time()
C_qmc = mc_qmc_C(n, halton(k, n)) # Réduction de variance par variable de controle
print('MC_qmc : {}'.format(time.time() - start_MC_qmc))

def plot_qmc(N, n):
    halton_ = halton(k, n * N)
    simus = []
    for i in range(N):
        print(halton_[i*n:(i + 1)*n])
        C = mc_qmc_C(n, halton_[i*n:(i + 1)*n])
        print(C)
        simus.append(C)
    return simus


def plot_everything(N, n):
    methods = [mc_C, mc_C_anti, mc_C_control]
    labels = ['Monte Carlo', 'Antithetic Variates', 'Control Variates', 'QMC']
    simus = [[method(n) for _ in range(N)] for method in methods]
    simu_qmc = plot_qmc(N, n)
    simus.append(simu_qmc)
    plt.boxplot(simus, labels=labels)
    plt.show()


#plot_everything(100, 100)
