import matplotlib.pyplot as plt
from mobspy import *
from assign import Assign
from math import ceil
import numpy as np
from functional_compiler import Jump_chain_qsd_bound
from rate_functions import *
import utils_rare_event as ure
from tqdm import tqdm


def find_unstable_equilibrium_point(p, K):
    Hl = [float(x) for x in np.linspace(0, p, 100)]

    b_d = sum_l(set_rate(Hl, P=p), amp_rate(Hl, K=K, P=p))
    b_d = sum_l(b_d, neg(reset_rate(Hl)))

    flag_f_zero = False
    for i, b in enumerate(b_d):
        if b <= 0:
            flag_f_zero = True
        if flag_f_zero and b > 0:
            eq_point = Hl[i]
            break

    eq_point = ceil(eq_point)
    return eq_point

def find_decay_parameter(p, K):

    L, H = BaseSpecies()

    L >> H[sigma]
    H >> L[p]
    L + H >> 2 * H[lambda l, h: k * l * h ** n / (h ** n + K ** n)]

    eq_point = find_unstable_equilibrium_point(p, K)

    L(p), H(0)
    S = Simulation(L | H)

    def H_from_L(h):
        return p - h

    interval = {H: [0, eq_point], L: Assign(H_from_L, H)}
    J = Jump_chain_qsd_bound(S, interval, verbose=False)
    _, decay = J.calculate_bound()
    return decay

latex_code = '\\begin{table}[h!]\n' + '\label{table:decay_parameters_vs_pop}\n' + '\\begin{tabular}{|l|l|l|l|l|}\n' \
             + '\hline\n' + 'P   & $K = 0.3 \cdot P$   & $K = 0.5 \cdot P$   & $P_d(T^- > 24\\text{h})$, $K = 0.3 ' \
                            '\cdot P$  &  $P_d(T^- > 24\\text{h})$, $K = 0.5 \cdot P$ \\\\\n'

# Parameters
Pop_values = [int(x) for x in np.linspace(10, 100, 10)]
decay_30 = {}
decay_50 = {}
prob_30 = {}
prob_50 = {}
for p in Pop_values:
    d_30 = find_decay_parameter(p, 0.3*p)
    decay_30[p] = d_30
    if not np.isinf(d_30):
        prob_30[p] = np.exp(-d_30*24)*100
    else:
        prob_50[p] = 100

    d_50 = find_decay_parameter(p, 0.5*p)
    decay_50[p] = d_50
    if not np.isinf(d_50):
        prob_50[p] = np.exp(-d_50*24)*100
    else:
        prob_50[p] = 100

    latex_line = f'\hline ${p}$ & ${ure.round_to_significant(d_30, 2)}$ & ${ure.round_to_significant(d_50, 2)}$ & ' \
                 f'${ure.round_to_significant(prob_30[p], 2, 1)}$ & ${ure.round_to_significant(prob_50[p], 2, 1)}$' \
                 f' \\\\\n'
    latex_code += latex_line

latex_code = latex_code + '\hline\n' + '\\end{tabular}\n' + '\\end{table}\n'

print(latex_code)
exit()

# print('Decay_30: ' + str(decay_30))
# print('Prob_30:' + str(prob_30))
# print('Decay_50: ' + str(decay_50))
# print('Prob_50: ' + str(prob_50))




