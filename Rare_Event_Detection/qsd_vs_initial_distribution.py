import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.linalg import expm
import rare_event_model_gen as rgen
import scipy.linalg as sclig
from parameters import *

# Function to calculate the distribution
def calculate_distribution(d, t):
    return np.matmul(d, sclig.expm(t * Q_full))

# Load the Q matrix and steady-state distributions
Q_full = rgen.return_re_model(full=True).q_matrix[:-1, :-1]
qsd_minus = rgen.return_re_model(reverse=False).calculate_qsd()
qsd_plus = rgen.return_re_model(reverse=True).calculate_qsd()

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

d_minus = np.zeros(len(Q_full))
d_minus[0] = 1
d_minus = calculate_distribution(d_minus, 1000)

# Plot 1
axes[0, 0].plot([i for i in range(len(Q_full))], d_minus, label=r'$d_{H0}$')
axes[0, 0].plot([i for i in range(len(qsd_minus))], qsd_minus, label=r'$\pi_{-}$')
axes[0, 0].set_title("Convergence H = 0")
axes[0, 0].set_xlabel("H (count)")
axes[0, 0].set_ylabel("State Probability")
axes[0, 0].legend()

# Plot 2
for t in [1, 1e4, 1e5, 1e6]:
    dif_minus = np.zeros(len(Q_full))
    d_minus = np.zeros(len(Q_full))
    d_minus[0] = 1
    d_minus = calculate_distribution(d_minus, t)

    for i in range(len(Q_full)):
        try:
            dif_minus[i] = abs(qsd_minus[i] - d_minus[i])
        except IndexError:
            dif_minus[i] = d_minus[i]
    axes[0, 1].plot([i for i in range(len(Q_full))], dif_minus, label=f't={t} (h)')

axes[0, 1].set_title(r"Difference $\pi_{-}$ and $d_{H0}$")
axes[0, 1].set_xlabel("H (count)")
axes[0, 1].set_ylabel("State Probability")

d_plus = np.zeros(len(Q_full))
d_plus[P] = 1
d_plus = calculate_distribution(d_plus, 1000)

# Plot 3
axes[1, 0].plot([i for i in range(len(Q_full))], d_plus, label=r'$d_{HP}$')
axes[1, 0].plot([P - i + 1 for i in reversed(range(len(qsd_plus)))], qsd_plus, label=r'$\pi_{+}$')
axes[1, 0].set_title("Convergence H = P")
axes[1, 0].set_xlabel("H (count)")
axes[1, 0].set_ylabel("State Probability")
axes[1, 0].legend()

# Plot 4
for t in [500, 5e9, 1e12]:
    dif_plus = np.zeros(len(Q_full))
    d_plus = np.zeros(len(Q_full))
    d_plus[-1] = 1
    d_plus = calculate_distribution(d_plus, t)

    for i in range(len(Q_full)):
        try:
            dif_plus[-i] = abs(qsd_plus[-i - 1] - d_plus[-i])
        except IndexError:
            dif_plus[-i] = d_plus[-i]
    axes[1, 1].plot([i for i in range(len(Q_full))], dif_plus, label=f't={t} (h)')

axes[1, 1].set_title(r"Difference $\pi_{+}$ and $d_{HP}$")
axes[1, 1].set_xlabel("H (count)")
axes[1, 1].set_ylabel("State Probability")

# Adjust layout
plt.tight_layout()

# Add legends to the subplots
for ax in axes.flat:
    ax.legend()

# Save figure
plt.savefig("figures/qsd_comparison.png")

plt.show()
