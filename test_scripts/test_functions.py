from mobspy import *
from functional_compiler import Jump_chain_qsd_bound
import matplotlib.pyplot as plt
import numpy as np

def idk():
    A, B = BaseSpecies()

    A + B >> A + 2*B [lambda r1, r2: r1*r2]
    B + A >> B + 2*A [lambda r1, r2: r1*r2]
    A >> Zero [1]
    B >> Zero [1]

    A(3), B(3)
    S = Simulation(A | B)
    interval = {A: [1, 5], B: [1, 5]}
    J = Jump_chain_qsd_bound(S, interval)
    J.calculate_bound()
idk()

def idk_2():
    A, B = BaseSpecies()

    A >> 2*A [1]
    A >> Zero [1]

    A(3)
    S = Simulation(A)
    interval = {A: [1, 5]}
    conditional_exit_bound = {A: '+'}
    J = Jump_chain_qsd_bound(S, interval, conditional_exit_bound)
    J.calculate_bound()
# idk_2()



