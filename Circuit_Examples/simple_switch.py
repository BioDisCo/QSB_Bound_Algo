from mobspy import *
import numpy as np
from functional_compiler import q_compile, q_matrix_generator, bound_estimator, Jump_chain_qsd_bound


initial_count, promoter_number = 60, 60
P1, P2, Mortal = BaseSpecies()
S1, S2 = New(Mortal)

Mortal >> Zero [1]

S2 + P1 >> S2 + P1 + S1[lambda s, p: p*1/(1 + (s/20)**4)]
S1 + P2 >> S1 + P2 + S2[lambda s, p: p*1/(1 + (s/20)**4)]

S1(initial_count), S2(0), P1(promoter_number), P2(promoter_number)
S = Simulation(S1 | S2 | P1 | P2)

def run_model():
    S.duration = 10
    S.run()
# run_model()

def bound_calculation():
    interval = {S1: [int(initial_count/2), int(1.5*initial_count)],
                S2: [0, int(initial_count/2)],
                P1: [promoter_number, promoter_number],
                P2: [promoter_number, promoter_number]}
    J = Jump_chain_qsd_bound(S, interval)
    J.calculate_bound()
bound_calculation()
