from mobspy import *
from functional_compiler import Jump_chain_qsd_bound
import matplotlib.pyplot as plt

A = BaseSpecies()

A >> 2*A [lambda r: (1000 - r)*r]
A >> Zero [1]

A(1)
interval = {A: [1, 2000]}
exit_direction = {A: 'below'}
S = Simulation(A)
J = Jump_chain_qsd_bound(S, interval=interval, exit_direction=exit_direction)
qsd = J.calculate_qsd()
plt.plot(qsd)
plt.show()
