from mobspy import *

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functional_compiler import Jump_chain_qsd_bound
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# I need assignments for this model

# Sim parameters
by = 43.8
bx = 36.2
Ki = 2.22
Kt = 27.4
Kl = 4.17
Ka = 0.133
dy = 1
dx = 1
nt = 2.29
na = 1.61
nl = 2.17
############
by = 43.8
bx = 36.2
Ki = 2.22
Kt = 27.4*5
Kl = 4.17*5
Ka = 0.133*0.05
dy = 1
dx = 1
nt = 2.29
na = 1.61
nl = 2.17
# 44 and 12
#######
by, bx, Ki, Kt, Kl, Ka, dy, dx, nt, na, nl = 43.8, 36.2, 2.22, 27.4*5, \
                                             4.17*5, 0.133*0.05, 1, 1, 2.29, 1.61, 2.17

LacI, TetR, IPTG, AHL = BaseSpecies()

# LacI repression and TetR decay
TetR >> TetR + LacI [lambda r: repression_rate(r, None, None, by, Kt, nt, None, None, None)]
TetR >> Zero [dx]

# TetR repression and LacI
rate_f = lambda r, r2, r3: repression_rate(r, r2, r3, bx, Kl, nl, Ki, Ka, na)
LacI + IPTG + AHL >> LacI + IPTG + AHL + TetR [rate_f]
LacI >> Zero [dy]

def repression_rate(r, r2, r3, b, K, n, K_I, K_ahl, n_ahl):
    if r2 is not None:
        r = r/(1 + K_I*r2)
    if r3 is not None:
        f = (K_ahl*r3)**n_ahl/(1 + (K_ahl*r3)**n_ahl)
    return b/(1 + (K*r)**n)*f if r3 is not None else b/(1 + (K*r)**n)

model = set_counts({LacI: 45, TetR: 0, IPTG: 0, AHL: 1})
Sim = Simulation(model)


# Starting from red initial state
def verify_hysteresis():
    results = []
    x = np.arange(0, 10, 0.1).tolist()
    for i in x:
        model = set_counts({LacI: 0, TetR: 50, IPTG: 0, AHL: i})
        S = Simulation(model)
        S.duration = 100
        S.plot_data = False
        S.level = -1
        S.run()
        print(S.fres['LacI'][-1], S.fres['TetR'][-1])
        results.append(S.fres['LacI'][-1])
    plt.plot(x, results)
    plt.show()
# verify_hysteresis()


def verify_irreversibility():
    model = set_counts({LacI: 0, TetR: 0, IPTG: 0, AHL: 0})
    S = Simulation(model)
    S.duration = 40
    S.plot_data = False
    S.level = -1
    with S.event_time(20):
        AHL(100)
    S.run()
    S.plot(LacI, TetR)
# verify_irreversibility()


def verify_switching():
    model = set_counts({LacI: 0, TetR: 50, IPTG: 0, AHL: 100})
    S = Simulation(model)
    S.duration = 40
    S.plot_data = False
    S.level = -1
    with S.event_time(20):
        AHL(0)
    S.run()
    S.plot(LacI, TetR)
# verify_switching()


def verify_steady_states():

    model = set_counts({LacI: 0, TetR: 50, IPTG: 0, AHL: 100})
    S = Simulation(model)
    S.duration = 40
    S.plot_data = False
    S.level = -1
    S.run()
    print(S.fres['LacI'][-1], S.fres['TetR'][-1])

    model = set_counts({LacI: 50, TetR: 0, IPTG: 0, AHL: 0})
    S = Simulation(model)
    S.duration = 40
    S.plot_data = False
    S.level = -1
    S.run()
    print(S.fres['LacI'][-1], S.fres['TetR'][-1])
# verify_steady_states()

# Hardest one to do
def bound_calculation(AHL_count=0):
    model = set_counts({LacI: 45, TetR: 0, IPTG: 0, AHL: 1})
    interval = {LacI: [200, 10], TetR: [0, 8], IPTG: [0, 0], AHL: [1, 1]}
    S = Simulation(model)
    # exit_direction = {LacI: 'bellow'}
    J = Jump_chain_qsd_bound(S, interval)
    J.calculate_bound()
# bound_calculation()

def partial_qsd_calculation():
    model = set_counts({LacI: 45, TetR: 0, IPTG: 0, AHL: 1})
    interval = {LacI: [200, 10], TetR: [0, 8], IPTG: [0, 0], AHL: [1, 1]}
    S = Simulation(model)
    # exit_direction = {LacI: 'bellow'}
    J = Jump_chain_qsd_bound(S, interval)
    qsd = J.partial_qsd(LacI)
    plt.plot([_ for _ in range(10, 200 + 1)], qsd)

    plt.title('Partial QSD of LacI')
    plt.ylabel('LacI State Probability')
    plt.xlabel('LacI (counts)')

    plt.savefig('figures/yolanda_partial_qsd.png')
    plt.show()


def leaving_probability_data():
    model = set_counts({LacI: 45, TetR: 0, IPTG: 0, AHL: 1})
    S = Simulation(model)
    S.method = 'directmethod'
    S.duration = 1/7.453531375456242e-09
    S.repetitions = 1000
    S.plot_data = False
    S.run()

    # Cannot be simulated error

    with open('figures/yolanda_switch.pkl', 'wb') as f:
        pkl.dump(Sim.results, f)


def compare_probabilities():

    with open('figures/yolanda_switch.pkl', 'rb') as f:
        res = pkl.load(f)

    positive_run = 0
    for Ll, Lt in zip(res[LacI], res[TetR]):

        for l, t in zip(Ll, Lt):
            if l < 10 or l > 200:
                positive_run += 1
                break
            if t > 8:
                positive_run += 1
                break

    print(positive_run/len(res[LacI]))
    print(np.exp(1/7.453531375456242e-09*-7.453531375456242e-09))


if __name__ == '__main__':
    pass
    # verify_hysteresis()
    # verify_irreversibility()
    # verify_switching()
    # verify_steady_states()
    bound_calculation()
    # leaving_probability_data()
    # compare_probabilities()
    # partial_qsd_calculation()
