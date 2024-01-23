from mobspy import *
from functional_compiler import q_compile, q_matrix_generator, bound_estimator, Jump_chain_qsd_bound

# @TODO test multiple functions - does it hold?

A = BaseSpecies()

# A + B >> Zero [0.0001]
# A + B >> Zero [0.001/u.h]
A >> 2*A [10]
A >> Zero [10]

A(15)
S = Simulation(A)
interval = {A: [10, 20]}
Jump_chain_qsd_bound(S, interval)





