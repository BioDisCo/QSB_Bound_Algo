import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.linalg import expm
import rare_event_model_gen as rgen
import scipy.linalg as sclig
from parameters import *
import utils_rare_event as ure


eq_point = rgen.find_unstable_equilibrium_point(P, K)


latex_code = ''

sgv_l = [1.8, 2, 3, 5]
for s in sgv_l:

    Q_full = rgen.return_re_model(sigma=s, full=True).q_matrix[:-1, :-1]
    d = np.zeros(len(Q_full))
    d[0] = 1
    d = np.matmul(d, sclig.expm(1 * Q_full))

    result = ure.round_to_significant(sum(d[eq_point:]), 3)

    latex_code += f'\hline ${s}$ & ${result}$ \\\\\n'
print(latex_code)



