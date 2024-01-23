from mobspy import *
from functional_compiler import Jump_chain_qsd_bound
import matplotlib.pyplot as plt
import numpy as np

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

LacI, TetR, IPTG, AHL = BaseSpecies()

# LacI repression and TetR decay
TetR >> TetR + LacI [lambda r: repression_rate(r, None, None, by, Kt, nt, None, None, None)]
TetR >> Zero [dx]

# TetR repression and LacI
LacI + IPTG + AHL >> LacI + IPTG + AHL + TetR [lambda r, r2, r3: repression_rate(r, r2, r3, bx, Kl, nl, Ki, Ka, na)]
LacI >> Zero [dy]

def repression_rate(r, r2, r3, b, K, n, K_I, K_ahl, n_ahl):
    if r2 is not None:
        r = r/(1 + K_I*r2)
    if r3 is not None:
        f = (K_ahl*r3)**n_ahl/(1 + (K_ahl*r3)**n_ahl)
    return b/(1 + (K*r)**n)*f if r3 is not None else b/(1 + (K*r)**n)

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
    interval = {LacI: [100, 10], TetR: [0, 8], IPTG: [0, 0], AHL: [1, 1]}
    S = Simulation(model)
    exit_direction = {LacI: 'bellow'}
    J = Jump_chain_qsd_bound(S, interval, exit_direction)
    J.calculate_bound()
# bound_calculation()

if __name__ == '__main__':
    pass
    # verify_hysteresis()
    # verify_irreversibility()
    # verify_switching()
    # verify_steady_states()
    bound_calculation()
