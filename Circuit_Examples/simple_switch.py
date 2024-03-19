import matplotlib.pyplot as plt
from mobspy import *
import numpy as np
from functional_compiler import q_compile, q_matrix_generator, bound_estimator, Jump_chain_qsd_bound
import pickle as pkl
import utils_rare_event as ure

f_1, f_2, f_3 = 2, 0, 1/3
initial_count, promoter_number = 30, 30
P1, P2, Mortal = BaseSpecies()
S1, S2 = New(Mortal)

Mortal >> Zero [0.001]

S2 + P1 >> S2 + P1 + S1[lambda s, p: 0.001*p*1/(1 + (s/10)**4)]
S1 + P2 >> S1 + P2 + S2[lambda s, p: 0.001*p*1/(1 + (s/10)**4)]

S1(initial_count), S2(0), P1(promoter_number), P2(promoter_number)
Sim = Simulation(S1 | S2 | P1 | P2)

def run_model():
    Sim.duration = 10
    Sim.run()
# run_model()

def bound_calculation(initial_state):
    S1(initial_state), S2(0), P1(promoter_number), P2(promoter_number)

    interval = {S1: [0, int(initial_count * f_1)],
                S2: [0, int(initial_count * f_3)],
                P1: [promoter_number, promoter_number],
                P2: [promoter_number, promoter_number]}
    exit_direction_s1 = {S1: 'below'}
    J = Jump_chain_qsd_bound(Sim, interval, exit_direction=exit_direction_s1)
    J.calculate_bound()
# bound_calculation(30)


def A_data_generation():
    results = []
    for i in [50, 30, 10, 0]:
        S1(i), S2(0), P1(promoter_number), P2(promoter_number)

        interval = {S1: [0, int(initial_count * f_1)],
                    S2: [0, int(initial_count * f_3)],
                    P1: [promoter_number, promoter_number],
                    P2: [promoter_number, promoter_number]}
        exit_direction_s1 = {S1: 'below'}
        J = Jump_chain_qsd_bound(Sim, interval, exit_direction=exit_direction_s1)
        A, decay = J.calculate_bound(probability_epsilon_bound=1e-7, stopping_probability=1e-7)
        results.append((A, decay))

    with open('figures/A_decay_res.pkl', 'wb') as f:
        pkl.dump(results, f)

    with open('figures/A_decay_res.pkl', 'rb') as f:
        res = pkl.load(f)
    print(res)
# A_data_generation()


def table_gen_from_A_data():
    latex_code = '\\begin{table}[h!]\n' + '\centering\n' + '\label{table:A_value_switch}\n' + \
                 '\\begin{tabular}{|l|l|}\n' \
                 + '\hline\n' + 'Initial Value S1 & A \\\\\n'

    with open('figures/A_decay_res.pkl', 'rb') as f:
        res = pkl.load(f)
    for i, A in zip([50, 30, 10, 0], res):
        latex_code = latex_code + f'\hline$({i}, 0)$ & ${ure.round_to_significant(A[0], 2)}$ \\\\\n'
    latex_code = latex_code + '\hline\n' + '\\end{tabular}\n' + '\\end{table}\n'
    print(latex_code)
    print()
    print('Decay: ' + str(res[0][1]))
# table_gen_from_A_data()


def qsd_prediction():
    interval = {S1: [0, int(initial_count*f_1)],
                S2: [0, int(initial_count*f_3)],
                P1: [promoter_number, promoter_number],
                P2: [promoter_number, promoter_number]}
    exit_direction_s1 = {S1: 'below'}

    S1(initial_count), S2(0), P1(promoter_number), P2(promoter_number)
    Sim1 = Simulation(S1 | S2 | P1 | P2)

    J = Jump_chain_qsd_bound(Sim1, interval, exit_direction=exit_direction_s1)
    partial = J.partial_qsd(S1)
    plt.plot([i for i in range(0, int(f_1*initial_count) + 1)], partial, label='High S1 QSD')

    S1(0), S2(initial_count), P1(promoter_number), P2(promoter_number)
    Sim2 = Simulation(S1 | S2 | P1 | P2)

    interval = {S1: [0, int(initial_count*f_3)],
                S2: [0, int(f_1*initial_count)],
                P1: [promoter_number, promoter_number],
                P2: [promoter_number, promoter_number]}
    exit_direction_s2 = {S2: 'below'}

    J = Jump_chain_qsd_bound(Sim2, interval, exit_direction=exit_direction_s2)
    partial = J.partial_qsd(S1)
    plt.plot([i for i in range(0,  int(f_3*initial_count) + 1)], partial, label='Low S1 QSD')

    plt.title('QSDs of the bistable switch BCRN')
    plt.xlabel('S1 state probability')
    plt.ylabel('S1 (counts)')
    plt.legend()
    plt.savefig("figures/QSD_Switch.png")
    plt.show()
# qsd_prediction()


def qsd_comparison():
    total_interval = {S1: [0, int(initial_count*f_1)],
                      S2: [0, int(initial_count*f_1)],
                      P1: [promoter_number, promoter_number],
                      P2: [promoter_number, promoter_number]}
    exit_direction = {S1: 'below', S2: 'below'}

    Jo = Jump_chain_qsd_bound(Sim, total_interval, exit_direction=exit_direction)
    Q = Jo.q_matrix[:-1, :-1]
    d = np.zeros(len(Q))
    for state in Jo.state_to_index:
        if state[2] == 30 and state[3] == 0:
            d[Jo.state_to_index[state]] = 1
            break
    d = ure.calculate_distribution(d, 100000, Q)

    interval = {S1: [0, int(initial_count*f_1)],
                S2: [0, int(initial_count*f_3)],
                P1: [promoter_number, promoter_number],
                P2: [promoter_number, promoter_number]}
    exit_direction = {S1: 'below'}

    J1 = Jump_chain_qsd_bound(Sim, interval, exit_direction=exit_direction)
    qsd = J1.calculate_qsd()

    distance = 0
    x = []
    y = []
    for state, io in Jo.state_to_index.items():
        try:
            i1 = J1.state_to_index[state]
            d_step = abs(qsd[i1] - d[io])
            distance += d_step
        except KeyError:
            d_step = abs(d[io])
            distance += d_step
        x.append(io)
        y.append(d_step)
    print(distance)
qsd_comparison()

def Gillespiere_Comparison():

    t = 1/4.3820718429521486e-07
    Sim.duration = t
    Sim.method = 'directmethod'
    Sim.repetitions = 1000
    Sim.plot_data = False
    Sim.run()

    with open('figures/switch_sim_data.pkl', 'wb') as f:
        pkl.dump(Sim.results, f)
# Gillespiere_Comparison()


def Compare_Probabilities():

    with open('figures/switch_sim_data.pkl', 'rb') as f:
        res = pkl.load(f)

    positive_run = 0
    for r in res[S2]:
        for v in r:
            if v > initial_count*f_3:
                positive_run += 1
                break

    print(positive_run/len(res[S2]))
    print(np.exp(1/4.3820718429521486e-07*-4.3820718429521486e-07))
# Compare_Probabilities()


