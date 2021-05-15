import numpy as np
import math
import random
import matplotlib.pyplot as plt
import sobol_seq
from scipy.stats import norm

# Let's set our seed

np.random.seed(777)

# Fixons nos constantes

alpha = 0.2
b = 0
sigma = 0.3
T = 1
r = 0.05
K = 5
k = 20
t = lambda i : i/20
S_0 = 20

# Simulons un mouvement brownien - source : https://towardsdatascience.com/animated-visualization-of-brownian-motion-in-python-3518ecf28533

def brownian_motion(N=50, T=1, h=1):
    dt = 1. * T/N  # the normalizing constant
    random_increments = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)  # the epsilon values
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    brownian_motion = np.insert(brownian_motion, 0, 0.0) # insert the initial condition
    
    return brownian_motion

# Simulons le modèle CIR : https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0
# St+1 - St = delta * alpha * (b - St) + sigma * sqrt(St) * epsilont 
# TODO check cir why negative 

def cir(S_0=S_0, alpha=alpha, b=b, sigma=sigma, k=k, T=T):
    dt = T/float(k)
    spots = [S_0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * np.random.normal()
        spots.append(spots[-1] + ds)
    return spots

def anti_cir(S_0=S_0, alpha=alpha, b=b, sigma=sigma, k=k, T=T):
    dt = T/float(k)
    spots = [S_0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * - np.random.normal()
        spots.append(spots[-1] + ds)
    return spots

# Fonctions de simulation de C

def phi(S, r=r, T=T, k=k, K=K):
    return math.exp(-r * T) * max(1/k * sum(S) - K, 0)

mc_C = lambda n : 1/n * sum([phi(cir()) for _ in range(n)])

def mc_C_anti(n):
    su = [phi(cir()) + phi(anti_cir()) for _ in range(n)]
    return 1/(2 * n) * sum(su)

def mc_C_control(n):
    Z = np.random.normal(size=n)
    phi_X = [phi(cir()) for _ in range(n)]
    beta = np.cov(phi_X, Z)[0][1]

    return 1/n * sum([phi_X[_] - beta * Z[_] for _ in range(n)])

# Simulations 

n = 100

C = mc_C(n) # Simulation classique de C
C_anti = mc_C_anti(n) # Réduction de variance par variable antithétique
C_control = mc_C_control(n) # Réduction de variance par variable de controle

# Comparons les erreures de Monte-Carlo

def plot_mcerr(methods, N, n, labels):
    simus = [[method(n) for _ in range(N)] for method in methods]
    plt.boxplot(simus, labels=labels)
    plt.show()

#plot_mcerr([mc_C, mc_C_anti, mc_C_control], 100, 100, labels=['Monte Carlo', 'Antithetic Variates', 'Control Variates'])

# TODO faire varier les paramêtres et le pas de discrétisation, observer résultats

# Multi-level Monte-Carlo

M = 5
e = math.e ** -10
L = int(math.log(e ** -1) / math.log(M))
N_0 = 100

brownian = lambda l : [np.random.normal() for _ in range(M ** l)]

def cir_mlmc(l, brownian, S_0=S_0, alpha=alpha, b=b, sigma=sigma, M=M, T=T):
    dt = M ** -l * T
    spots = [S_0]
    for i in range(M ** l):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * brownian[i]
        spots.append(spots[-1] + ds)
    return spots


h = lambda l : M ** -l * T
V = lambda l : np.var(cir_mlmc(l, brownian(l)))
P = lambda l, brownian : phi(cir_mlmc(l, brownian))
N = lambda l : int(100 * (V(l) * h(l)) ** 0.5)


def brownian_bis(brownian):
    return [sum(brownian[i:i+M]) for i in range(len(brownian)//M)]

def Y(l):
    su = 0
    print('NL : ' + str(N(l)))
    for i in range(N(l)):
        bro = brownian(l)
        bbro = brownian_bis(bro)
        p = P(l, bro) - P(l-1, bbro)
        su += p
    return 1/N(l) * su

#Y_0 = 1/N_0 * sum(P(0, brownian(0)) for _ in range(N_0))
#Y = Y_0 + sum(Y(l) for l in range(1, L))

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
# Génération séquence sobol normale

def i4_sobol_generate_std_normal(dim_num, n, skip=0):
    """
    Generates multivariate standard normal quasi-random variables.
    Parameters:
      Input, integer dim_num, the spatial dimension.
      Input, integer n, the number of points to generate.
      Input, integer SKIP, the number of initial points to skip.
      Output, real np array of shape (n, dim_num).
    """

    sobols = sobol_seq.i4_sobol_generate(dim_num, n, skip)

    normals = norm.ppf(sobols)

    return normals

# Calculons option asiatique  


def cir_qmc(halt, S_0=S_0, alpha=alpha, b=b, sigma=sigma, k=k, T=T):
    dt = T/float(k)
    spots = [S_0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(dt * spots[-1]) * halt[i]
        spots.append(spots[-1] + ds)
    return spots

halton_ = halton(k, n)
mc_qmc_C = lambda n, halton_: 1/n * sum([phi(cir_qmc(halton_[i])) for i in range(n)])

def plot_qmc(N, n):
    halton_ = halton(k, n * N)
    simus = []
    for i in range(N):
        print(halton_[i*n:(i + 1)*n])
        C = mc_qmc_C(n, halton_[i*n:(i + 1)*n])
        print(C)
        simus.append(C)
    plt.boxplot(simus)
    plt.show()

# TODO plot tout dans le même graphe
