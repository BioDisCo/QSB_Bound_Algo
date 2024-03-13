import matplotlib.pyplot as plt

from rate_functions import *
import numpy as np
from scipy.linalg import expm

Q = Q_matrix(Hl, set_rate, reset_rate, amp_rate)

for i in [0, 30, 119]:

    isv = np.zeros((len(Hl)))   
    isv[i] = 1

    t = 100*24
    state_vector = np.matmul(isv, expm(t*Q))
    plt.plot(state_vector)

plt.show()



