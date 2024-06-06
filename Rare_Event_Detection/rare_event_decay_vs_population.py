from rate_functions import *
import utils_rare_event as ure
import rare_event_model_gen as rgen

def false_ratio(reverse=False):

    latex_code = ''
    # Pop_values = [int(x) for x in np.linspace(20, 150, 10)]
    Pop_values = [20, 60, 100, 140, 180]
    decay_30 = {}
    decay_50 = {}
    prob_30 = {}
    prob_50 = {}
    for pop in Pop_values:
        d_30 = rgen.return_re_model(pop=pop, K=0.3*pop, reverse=reverse).calculate_bound()[1]
        decay_30[pop] = d_30
        if not np.isinf(d_30):
            prob_30[pop] = np.exp(-d_30*24)*100
        else:
            prob_50[pop] = 100

        d_50 = rgen.return_re_model(pop=pop, K=0.5*pop, reverse=reverse).calculate_bound()[1]
        decay_50[pop] = d_50
        if not np.isinf(d_50):
            prob_50[pop] = np.exp(-d_50*24)*100
        else:
            prob_50[pop] = 100

        latex_line = f'\hline ${pop}$ & ${ure.round_to_significant(d_30, 2)}$ & ' \
                     f'${ure.round_to_significant(d_50, 2)}$ & ' \
                     f'${ure.round_to_significant(prob_30[pop], 2, 1)}$ &' \
                     f' ${ure.round_to_significant(prob_50[pop], 2, 1)}$' \
                     f' \\\\\n'
        latex_code += latex_line

    print(latex_code)
    return latex_code
false_ratio(reverse=True)

def find_decays_specific():
    pop = 150
    f = 0.4

    d_b = rgen.return_re_model(pop=pop, K=f * pop).calculate_bound()[1]
    d_f = rgen.return_re_model(pop=pop, K=f * pop, reverse=True).calculate_bound()[1]
    p_b = np.exp(-d_b * 24) * 100
    p_f = np.exp(-d_f * 24) * 100

    print(f'Decay Back: {d_b}, Decay Foward: {d_f}, P_b: {p_b}, P_f: {p_f}')
# find_decays_specific()



