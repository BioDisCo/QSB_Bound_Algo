import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rate_functions import *
from utils_rare_event import round_to_significant as ure_round_to_significant
import rare_event_model_gen as rgen

def false_ratio(reverse=False):

    latex_code = ''
    # Pop_values = [int(x) for x in np.linspace(20, 150, 10)]
    Pop_values = [20, 60, 100, 140, 200, 400, 750]
    decay_30 = {}
    decay_50 = {}
    prob_30 = {}
    prob_50 = {}
    for pop in Pop_values:
        d_30 = rgen.return_re_model(pop=pop, K=0.3*pop, reverse=reverse).calculate_bound()[1]
        decay_30[pop] = d_30
        if not np.isinf(d_30):
            prob_30[pop] = np.exp(-d_30*24)
        else:
            prob_30[pop] = 1

        d_50 = rgen.return_re_model(pop=pop, K=0.5*pop, reverse=reverse).calculate_bound()[1]
        decay_50[pop] = d_50
        if not np.isinf(d_50):
            prob_50[pop] = np.exp(-d_50*24)
        else:
            prob_50[pop] = 1


        latex_line = fr'\hline ${pop}$ & ${ure_round_to_significant(d_30)}$ & ' \
                     fr'${ure_round_to_significant(d_50)}$ & ' \
                     fr'${ure_round_to_significant(1 - prob_30[pop])}$ &' \
                     fr' ${ure_round_to_significant(1 - prob_50[pop])}$' \
                     f'\\\\\n'
        latex_code += latex_line

    print(latex_code)
    return latex_code

def find_decays_specific():
    pop = 150
    f = 0.4

    d_b = rgen.return_re_model(pop=pop, K=f * pop).calculate_bound()[1]
    d_f = rgen.return_re_model(pop=pop, K=f * pop, reverse=True).calculate_bound()[1]
    p_b = np.exp(-d_b * 24)
    p_f = np.exp(-d_f * 24)

    print(f'Decay Back: {d_b}, Decay Foward: {d_f}, P_b: {p_b}, P_f: {p_f}')
# find_decays_specific()


if __name__ == '__main__':
    false_ratio(reverse=True)


