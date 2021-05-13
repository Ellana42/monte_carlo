import numpy as np
import math
import random
import matplotlib.pyplot as plt

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
s0 = 20

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

def cir(s0=s0, alpha=alpha, b=b, sigma=sigma, k=k, T=T):
    dt = T/float(k)
    spots = [s0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(spots[-1]) * np.random.normal()
        spots.append(spots[-1] + ds)
    return spots

# Fonctions de simulation de C

def phi(S, r=r, T=T, k=k, K=K):
    return math.exp(-r * T) * max(1/k * sum(S) - K, 0)

mc_C = lambda n : 1/n * sum([phi(cir()) for _ in range(n)])

def mc_C_anti(n):
    su = [phi(cir()) + phi([-cir for cir in cir()]) for _ in range(n)]
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

#plot_mcerr([mc_C, mc_C_control], 100, 100, labels=['Monte Carlo', 'Control Variates'])

# TODO faire varier les paramêtres et le pas de discrétisation, observer résultats

# Multi-level Monte-Carlo

M = 5
N_ = 500
e = math.e ** -10
L = math.log(e ** -1) / math.log(M) 

def cir_mlmc(l, brownian, s0=s0, alpha=alpha, b=b, sigma=sigma, M=M, T=T):
    dt = M ** -l * T
    spots = [s0]
    for i in range(M ** l):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(spots[-1]) * np.random.normal()
        print(ds)
        spots.append(spots[-1] + ds)
    return spots

def P(l, brownian):
    while True : 
        try:
            return phi(cir_mlmc(l, brownian))
        except ValueError:
            continue

h = lambda l : M ** -l * T
V = lambda l : np.var(cir_mlmc(l)) 
#P = lambda l : phi(cir_mlmc(l))
N = lambda l : 23 * (V(l) * h(l)) ** 0.5

brownian = lambda l : [np.random.normal() for _ in range(M ** l)]

def brownian_bis(brownian):
    return [sum(brownian[i:i+M]) for i in range(len(brownian)//M)]


Y = lambda l, brownian : 1/N(l) * sum(P(l, brownian) - P(l-1, brownian_bis(brownian)) for i in range(N(l)))

