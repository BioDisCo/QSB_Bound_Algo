import rare_event_model_gen as rgen
import matplotlib.pyplot as plt
import scipy.linalg as sclig
import numpy as np
from rate_functions import *

# The equilibrium point is 49
Q = rgen.return_re_model(full=True).q_matrix[:-1, :-1]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

n_variables = Q.shape[0]

def calculate_distribution(d, t):
    Q_dense = Q.toarray()  # Convert sparse to dense
    return np.matmul(d, sclig.expm(t * Q_dense))

# Subplot (a) - 1000h
for i in [1, 47, 49, 53, 200]:
    d = np.zeros(n_variables)
    d[i] = 1
    d = calculate_distribution(d, 1000)
    axes[0, 0].plot(d, label=rf'$H_0 = {i}$ count')

axes[0, 0].set_title(r"$\bf{a}$ Distribution After 1.00e+03 h", fontsize=16)
axes[0, 0].set_xlabel("H (count)", fontsize=14)
axes[0, 0].set_ylabel("State Probability", fontsize=14)
axes[0, 0].legend()
axes[0, 0].tick_params(axis='both', labelsize=10)

# Subplot (b) - 1e6 h
for i in [1, 47, 49, 53, 200]:
    d = np.zeros(n_variables)
    d[i] = 1
    d = calculate_distribution(d, 1000000)
    axes[0, 1].plot(d, label=rf'$H_0 = {i}$ count')

axes[0, 1].set_title(r"$\bf{b}$ Distribution After 1.00e+06 h", fontsize=16)
axes[0, 1].set_xlabel("H (count)", fontsize=14)
axes[0, 1].set_ylabel("State Probability", fontsize=14)
axes[0, 1].legend()
axes[0, 1].tick_params(axis='both', labelsize=10)

# Subplot (c) - Equilibrium points
b_d = sum_l(set_rate(Hl), amp_rate(Hl))
b_d = sum_l(b_d, neg(reset_rate(Hl)))

axes[1, 0].plot(Hl, b_d)
axes[1, 0].plot(Hl, zero_line(Hl))
axes[1, 0].set_title(r"$\bf{c}$ Equilibrium Point Estimation", fontsize=16)
axes[1, 0].set_xlabel("H", fontsize=14)
axes[1, 0].set_ylabel("Set Rate + Hold Rate - Reset Rate", fontsize=14)
axes[1, 0].axvline(x=7, color='gray', linestyle='--', linewidth=1)
axes[1, 0].axvline(x=47.5, color='gray', linestyle='--', linewidth=1)
axes[1, 0].axvline(x=139, color='gray', linestyle='--', linewidth=1)
axes[1, 0].tick_params(axis='both', labelsize=10)

# Subplot (d) - Equilibrium points with different sigma
b_d = sum_l(set_rate(Hl, s=3), amp_rate(Hl))
b_d = sum_l(b_d, neg(reset_rate(Hl)))

axes[1, 1].plot(Hl, b_d)
axes[1, 1].plot(Hl, zero_line(Hl))
axes[1, 1].set_title(r"$\bf{d}$ Equilibrium Point Estimation", fontsize=16)
axes[1, 1].set_ylabel("Set Rate + Hold Rate - Reset Rate", fontsize=14)
axes[1, 1].tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig("figures/initial_distribution.png", dpi=300)  # Added higher DPI for better quality
plt.show()