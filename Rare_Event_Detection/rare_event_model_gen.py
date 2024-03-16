import matplotlib.pyplot as plt
from mobspy import *
from assign import Assign
from math import ceil
import numpy as np
from functional_compiler import Jump_chain_qsd_bound
from rate_functions import *
import utils_rare_event as ure
from tqdm import tqdm

def find_unstable_equilibrium_point(p, K, plot_bd=False):
    Hl = [float(x) for x in np.linspace(0, p, 100)]

    b_d = sum_l(set_rate(Hl, P=p), amp_rate(Hl, K=K, P=p))
    b_d = sum_l(b_d, neg(reset_rate(Hl)))

    if plot_bd:
        plt.plot(Hl, b_d)
        plt.plot(Hl, zero_line(Hl))
        plt.show()

    flag_f_zero = False
    for i, b in enumerate(b_d):
        if b <= 0:
            flag_f_zero = True
        if flag_f_zero and b > 0:
            eq_point = Hl[i]
            break

    eq_point = ceil(eq_point)
    return eq_point


def return_re_model(sigma=sigma, p=p, k=k, K=K, n=n, pop=P, reverse=False, full=False, initial_dist=None):
    L, H = BaseSpecies()

    L >> H[sigma]
    H >> L[p]
    L + H >> 2 * H[lambda l, h: k * l * h ** n / (h ** n + K ** n)]

    if not full:
        eq_point = find_unstable_equilibrium_point(pop, K)
    else:
        eq_point = P
        reverse = False

    if initial_dist is None:
        if not reverse:
            L(pop), H(0)
        else:
            L(0), H(pop)
    else:
        H(initial_dist), L(pop - initial_dist)

    S = Simulation(L | H)

    def H_from_L(h):
        return pop - h

    if not reverse:
        interval = {H: [0, eq_point], L: Assign(H_from_L, H)}
    else:
        interval = {H: [eq_point + 1, pop], L: Assign(H_from_L, H)}
    J = Jump_chain_qsd_bound(S, interval, verbose=False)
    return J

