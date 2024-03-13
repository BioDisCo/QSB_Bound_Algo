from mobspy import *
from assign import Assign
from functional_compiler import q_compile, q_matrix_generator, bound_estimator, Jump_chain_qsd_bound

# Parameters
K = 50
P = 120

n = 4

k = 35 # h-1
p = 14 # h-1

sigma = 0.5


L, H = BaseSpecies()

L >> H [sigma]
H >> L [p]
L + H >> 2*H [lambda l, h: k*l*h**n/(h**n + K**n)]


L(P), H(0)
S = Simulation(L | H)

def H_from_L(h):
    return P - h

interval = {H: [0, 5], L: Assign(H_from_L, H)}
J = Jump_chain_qsd_bound(S, interval)
J.calculate_bound()



