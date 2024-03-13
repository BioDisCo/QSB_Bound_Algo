from mobspy import *
import numpy as np
from functional_compiler import q_compile, q_matrix_generator, bound_estimator, Jump_chain_qsd_bound

initial_count, promoter_number = 100, 25
Promoters, Mortal = BaseSpecies()
Promoters.active, Promoters.inactive
P1, P2 = New(Promoters)
S1, S2 = New(Mortal)

Mortal >> Zero[0.01]

Rev[4*S1 + P2.active >> P2.inactive] [10**(1/4), 1]
Rev[4*S2 + P1.active >> P1.inactive] [10**(1/4), 1]

P1.active >> P1 + S1[1/promoter_number]
P2.active >> P2 + S2[1/promoter_number]

S1(initial_count)
# S2(100)
P1(promoter_number), P2(promoter_number)
S = Simulation(S1 | S2 | P1 | P2)
S.duration = 500
S.run()






