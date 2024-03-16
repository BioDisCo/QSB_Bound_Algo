import rare_event_model_gen as rgen
import matplotlib.pyplot as plt
import scipy.linalg as sclig
import numpy as np
from rate_functions import *


# The equilibrium point is 49

Q = rgen.return_re_model(full=True).q_matrix[:-1, :-1]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

def calculate_distribution(d, t):
    return np.matmul(d, sclig.expm(t * Q))

for i in [1, 47, 49, 53, 200]:

    d = np.zeros(len(Q))
    d[i] = 1
    d = calculate_distribution(d, 1000)

    axes[0, 0].plot(d, label=rf'$H_0 = {i}$')

axes[0, 0].set_title(r"State Probability Distribution After $1000$h")
axes[0, 0].set_xlabel("H (count)")
axes[0, 0].set_ylabel("State Probability")
axes[0, 0].legend()

# Compare with time 1e7

for i in [1, 47, 49, 53, 200]:

    d = np.zeros(len(Q))
    d[i] = 1
    d = calculate_distribution(d, 1000000)

    axes[0, 1].plot(d, label=rf'$H_0 = {i}$')

axes[0, 1].set_title(r"State Probability Distribution After $1000000$h")
axes[0, 1].set_xlabel("H (count)")
axes[0, 1].set_ylabel("State Probability")
axes[0, 1].legend()

b_d = sum_l(set_rate(Hl), amp_rate(Hl))
b_d = sum_l(b_d, neg(reset_rate(Hl)))

axes[1, 0].plot(Hl, b_d)
axes[1, 0].plot(Hl, zero_line(Hl))
axes[1, 0].set_title(r"Equilibrium Point Estimation")
axes[1, 0].set_xlabel("H")
axes[1, 0].set_ylabel("Set Rate + Hold Rate - Reset Rate")
axes[1, 0].axvline(x=7, color='gray', linestyle='--', linewidth=1)
axes[1, 0].axvline(x=47.5, color='gray', linestyle='--', linewidth=1)
axes[1, 0].axvline(x=139, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig("figures/initial_distribution.png")
plt.show()


