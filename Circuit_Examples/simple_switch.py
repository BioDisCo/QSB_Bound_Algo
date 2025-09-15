import matplotlib.pyplot as plt
from mobspy import *
import numpy as np
import os
import sys

from sympy.core.assumptions_generated import beta_rules

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functional_compiler import q_compile, q_matrix_generator, bound_estimator, Jump_chain_qsd_bound
import pickle as pkl
import utils_rare_event as ure


def generate_mobspy_switch_model(initial_count, promoter_number= 30,
                                 hill_coef = 2, alpha =  0.01, beta = 0.0002,
                                 f_1 = 2, f_2 = 1/2, f_3 = 1/10, return_species = False):
    P1, P2, Mortal = BaseSpecies()
    S1, S2 = New(Mortal)


    Mortal >> Zero [beta]

    S2 + P1 >> S2 + P1 + S1[lambda s, p: alpha*p*1/(1 + (s/10)**hill_coef)]
    S1 + P2 >> S1 + P2 + S2[lambda s, p: alpha*p*1/(1 + (s/10)**hill_coef)]

    S1(initial_count), S2(0), P1(promoter_number), P2(promoter_number)
    Sim = Simulation(S1 | S2 | P1 | P2)

    interval = {S1: [int(initial_count * f_2), int(initial_count* f_1)],
                S2: [0, int(initial_count * f_3)],
                P1: [promoter_number, promoter_number],
                P2: [promoter_number, promoter_number]}
    exit_direction = {S1: 'below'}

    if return_species:
        return Sim, interval, exit_direction, [S1, S2]
    else:
        return Sim, interval, exit_direction


def run_model(inital_count = 30, promoter_number =30, duration = 10):
    Sim, _, _ = generate_mobspy_switch_model(initial_count=inital_count, promoter_number=promoter_number)
    Sim.duration = duration
    Sim.run()
# run_model()

def bound_calculation(initial_state):
    Sim, interval, exit_direction_s1 = generate_mobspy_switch_model(initial_count=initial_state)
    J = Jump_chain_qsd_bound(Sim, interval, exit_direction=exit_direction_s1)
    J.calculate_bound()
# bound_calculation(30)


def A_data_generation():
    results = []
    for i in [50, 30, 10, 0]:
        Sim, interval, exit_direction_s1 = generate_mobspy_switch_model(initial_count=i)

        J = Jump_chain_qsd_bound(Sim, interval, exit_direction=exit_direction_s1)
        A, decay = J.calculate_bound(probability_epsilon_bound=1e-10, stopping_probability=1e-10)
        results.append((A, decay))

    with open('figures/A_decay_res.pkl', 'wb') as f:
        pkl.dump(results, f)

    with open('figures/A_decay_res.pkl', 'rb') as f:
        res = pkl.load(f)
# A_data_generation()


def table_gen_from_A_data():
    latex_code = r'\begin{table}[h!]' + '\n' + r'\centering' + '\n' + r'\label{table:A_value_switch}' + '\n' + \
                 r'\begin{tabular}{|l|l|}' + '\n' \
                 + r'\hline' + '\n' + r'Initial Value S1 & A \\' + '\n'

    with open('figures/A_decay_res.pkl', 'rb') as f:
        res = pkl.load(f)
    for i, A in zip([50, 30, 10, 0], res):
        latex_code = latex_code + f'\\hline$({i}, 0)$ & ${ure.round_to_significant(A[0], 2)}$ \\\\\n'
    latex_code = latex_code + r'\hline' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n'
    print(latex_code)
    print()
    print('Decay: ' + str(res[0][1]))
# table_gen_from_A_data()


def qsd_prediction():
    initial_count = 30
    promoter_number = 30
    f_1 = 2
    f_2 = 0
    f_3 = 1/3

    # HIGH S1 QSD (V1 region)
    Sim1, interval_v1, exit_direction_s1, [S1, S2] = generate_mobspy_switch_model(
        initial_count=initial_count,
        promoter_number=promoter_number,
        alpha=0.001, beta=0.001,
        hill_coef=4,
        f_1=f_1, f_2=f_2, f_3=f_3,
        return_species = True
    )

    J1 = Jump_chain_qsd_bound(Sim1, interval_v1, exit_direction=exit_direction_s1)

    partial_high_s1 = J1.partial_qsd(S1)
    partial_s2 = J1.partial_qsd(S2)
    plt.plot(range(len(partial_high_s1)), partial_high_s1, label='S1 QSD')
    plt.plot(range(len(partial_s2)), partial_s2, label='S2 QSD')


    plt.title('QSDs of the bistable switch BCRN', fontsize=16)
    plt.xlabel('Counts', fontsize=14)
    plt.ylabel('State probability', fontsize=14)
    plt.legend()
    plt.savefig("figures/QSD_Switch.png")
    plt.show()



def qsd_comparison():
    initial_count = 30
    promoter_number = 30

    # Create the full system (both regions combined)
    Sim_full, total_interval, exit_direction = generate_mobspy_switch_model(
        initial_count=initial_count,
        promoter_number=promoter_number,
        alpha=0.001, beta=0.001,
        hill_coef=4,
        f_1=2, f_2=0, f_3=2 # Full space: S1=[0, 60], S2=[0, 60]
    )

    Jo = Jump_chain_qsd_bound(Sim_full, total_interval, exit_direction=exit_direction)
    Q = Jo.q_matrix[:-1, :-1]

    # Initialize distribution at (30, 0) state

    d = np.zeros(max(Q.shape))
    for state in Jo.state_to_index:
        if state[2] == 30 and state[3] == 0:  # (S1=30, S2=0)
            d[Jo.state_to_index[state]] = 1
            break

    # Evolve the distribution
    d = ure.calculate_distribution(d, 100000, Q)

    # Create the cut process for V1 region
    Sim_cut, interval, exit_direction = generate_mobspy_switch_model(
        initial_count=initial_count,
        promoter_number=promoter_number,
        alpha=0.001, beta=0.001,
        hill_coef=4,
        f_1=2, f_2=0, f_3=1/3  # V1: S1=[0, 60], S2=[0, 10]
    )

    J1 = Jump_chain_qsd_bound(Sim_cut, interval, exit_direction=exit_direction)
    qsd = J1.calculate_qsd()

    # Compare the distributions
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

    plt.plot(x, y)
    plt.xlabel('State Index')
    plt.ylabel('Probability Difference')
    plt.title('QSD vs Time Evolution Difference')
    plt.show()
    print(f"Total variation distance: {distance}")


def Gillespiere_Comparison():

    Sim, _, _ = generate_mobspy_switch_model()

    t = 1/4.3820718429521486e-07
    Sim.duration = t
    Sim.method = 'directmethod'
    Sim.repetitions = 1000
    Sim.plot_data = False
    Sim.run()

    with open('figures/switch_sim_data.pkl', 'wb') as f:
        pkl.dump(Sim.results, f)
# Gillespiere_Comparison()


#def Compare_Probabilities():


    #with open('figures/switch_sim_data.pkl', 'rb') as f:
    #    res = pkl.load(f)

    #positive_run = 0
    #for r in res[S2]:
    #    for v in r:
    #        if v > initial_count*f_3:
    #            positive_run += 1
    #            break

    # print(positive_run/len(res[S2]))
    # print(np.exp(1/4.3820718429521486e-07*-4.3820718429521486e-07))
# Compare_Probabilities()


def population_sweep_all_hills():
    """Test all Hill coefficients in one go"""

    P_values = [2, 6, 10, 20, 40]  # Promoter levels to control proteins
    hill_coefficients = [2, 3, 4]
    test_times = [6 * 3600, 12 * 3600, 24 * 3600]  # 6h, 12h, 24h in seconds

    alpha = 0.05
    beta = 0.01

    all_results = []
    os.makedirs('figures', exist_ok=True)

    for n_h in hill_coefficients:
        for P in P_values:
            initial_S1 = int(alpha / beta * P)
            print(f"Testing P = {P}, n = {n_h}...")

            Sim, interval, exit_direction = generate_mobspy_switch_model(
                initial_count=initial_S1,
                promoter_number=P,
                alpha=alpha,
                beta=beta,
                hill_coef=n_h,
                f_1=2, f_2=1 / 3, f_3=1/5
            )

            J = Jump_chain_qsd_bound(Sim, interval, exit_direction=exit_direction)
            A, decay = J.calculate_bound(probability_epsilon_bound=1e-7,
                                         stopping_probability=1e-7)

            if decay > 0:
                # Calculate transition probabilities for different times
                probs = {}
                for t in test_times:
                    prob_bound = np.exp(-decay * t) - A
                    prob_bound = max(0, prob_bound)
                    transition_prob = 1 - prob_bound
                    probs[f'{int(t / 3600)}h'] = transition_prob

                mean_time = 1 / decay  # This is 1/λ
            else:
                probs = {f'{int(t / 3600)}h': 0 for t in test_times}
                mean_time = float('inf')

            all_results.append({
                'P': P,
                'n_h': n_h,
                'decay': decay,
                'A': A,
                'P_transition_6h': probs['6h'],
                'P_transition_12h': probs['12h'],
                'P_transition_24h': probs['24h'],
                'mean_time_1_over_lambda': mean_time,
                'mean_time_readable': format_time(mean_time)
            })


    # Generate table with multiple time points
    latex_code = '''\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|l|l|l|l|l|l|l|}
\\hline
$P$ & $n$ & Decay Parameter $(\\lambda)$ & $A$ & $\\mathbb{P}(\\text{transition} \\leq 6\\text{h})$ & $\\mathbb{P}(\\text{transition} \\leq 12\\text{h})$ & $\\mathbb{P}(\\text{transition} \\leq 24\\text{h})$ & $1/\\lambda$ \\\\
\\hline
'''

    for r in all_results:
        if isinstance(r['decay'], (float, np.float64)):
            latex_code += f"${r['P']}$ & ${r['n_h']}$ & ${ure.round_to_significant(r['decay'], 2)}$ & "
            latex_code += f"${ure.round_to_significant(r['A'], 2)}$ & "
            latex_code += f"${ure.round_to_significant(r['P_transition_6h'], 2)}$ & "
            latex_code += f"${ure.round_to_significant(r['P_transition_12h'], 2)}$ & "
            latex_code += f"${ure.round_to_significant(r['P_transition_24h'], 2)}$ & "
            latex_code += f"{r['mean_time_readable']} \\\\\n"
        else:
            latex_code += f"${r['P']}$ & ${r['n_h']}$ & Error & Error & Error & Error & Error & Error \\\\\n"

    latex_code += '''\\hline
\\end{tabular}
\\caption{Toggle Switch Robustness Analysis. Transition probabilities are calculated as $1 - \\max(0, e^{-\\lambda t} - A)$ for different time intervals. The last column shows $1/\\lambda$ which represents the characteristic time scale of the QSD.}
\\label{table:hill_comparison}
\\end{table}
'''

    with open('figures/hill_coefficient_sweep.pkl', 'wb') as f:
        pkl.dump(all_results, f)

    print(latex_code)


def format_time(seconds):
    """Convert seconds to readable format"""
    if seconds == float('inf'):
        return "∞"
    elif seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    elif seconds < 31536000:
        return f"{seconds / 86400:.1f} days"
    else:
        return f"{seconds / 31536000:.1f} years"


def generate_hill_comparison_table(reculate=True):
    """Generate LaTeX table comparing different Hill coefficients"""
    if reculate:
        population_sweep_all_hills()

    with open('figures/hill_coefficient_sweep.pkl', 'rb') as f:
        all_results = pkl.load(f)

    # Generate one combined table with consistent units (hours)
    latex_code = '''\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|l|l|l|l|l|}
\\hline
P & n & Decay Parameter & A & P(transition in 24h) & Mean Time (hours) \\\\
\\hline
'''

    for r in all_results:
        if isinstance(r['decay'], (float, np.float64)):
            # Convert mean time to hours consistently
            mean_time_hours = r['mean_time_sec'] / 3600 if r['mean_time_sec'] != float('inf') else float('inf')

            if mean_time_hours == float('inf'):
                mean_time_str = '$\\infty$'
            else:
                mean_time_str = f'${ure.round_to_significant(mean_time_hours, 2)}$'

            latex_code += f"${r['P']}$ & ${r['n_h']}$ & ${ure.round_to_significant(r['decay'], 2)}$ & "
            latex_code += f"${ure.round_to_significant(r['A'], 2)}$ & "
            latex_code += f"${ure.round_to_significant(r['P_transition_24h'], 2)}$ & "
            latex_code += f"{mean_time_str} \\\\\n"
        else:
            latex_code += f"${r['P']}$ & ${r['n_h']}$ & Error & Error & Error & Error \\\\\n"

    latex_code += '''\\hline
\\end{tabular}
\\caption{Toggle Switch Robustness vs Hill Coefficient and Promoter Copy Number}
\\label{table:hill_comparison}
\\end{table}
'''

    print(latex_code)


# Run the analysis
if __name__ == '__main__':
    population_sweep_all_hills()
    # qsd_prediction()
    # qsd_comparison()



