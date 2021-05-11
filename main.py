import numpy as np
import math
import random

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

#W = brownian_motion(N=1000)

# Simulons le modèle CIR : https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0
# St+1 - St = delta * alpha * (b - St) + sigma * sqrt(St) * epsilont 

def cir(s0=s0, alpha=alpha, b=b, sigma=sigma, k=k, T=T):
    dt = T/float(k)
    spots = [s0]
    for i in range(k):
        ds = alpha * (b - spots[-1]) * dt + sigma * math.sqrt(spots[-1]) * np.random.normal()
        spots.append(spots[-1] + ds)
    return spots

# Simulons C

def phi(CIR, r=r, T=T, k=k, K=K):
    return math.exp(-r * T) * max(1/k * sum(CIR) - K, 0)

def phi_anti(CIR, r=r, T=T, k=k, K=K):
    return math.exp(-r * T) * max(- 1/k * sum(CIR) - K, 0)
sample_size = 100

#C_li = [phi(cir()) for _ in range(sample_size)]
C = 1/sample_size * sum([phi(cir()) for _ in range(sample_size)])

# Réduction de variance par variable antithétique

#anti = [phi(cir()) +  phi_anti(cir()) for _ in range(sample_size)]
C_anti = 1/(2 * sample_size) * sum([phi(cir()) + phi_anti(cir()) for _ in range(sample_size)])

# Réduction de variance par variable de controle
