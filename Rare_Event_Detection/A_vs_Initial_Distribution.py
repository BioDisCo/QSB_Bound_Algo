import rare_event_model_gen as rgen
from parameters import *
import scipy.linalg as sclig
import numpy as np
from ..utils_rare_event import round_to_significant as ure_round_to_significant



latex_line = '\\begin{table}[h] \n\\label{table:A_vs_L1_qsd}\n\\begin{tabular}{|l|l|l|}\n\hline\n' \
             + 'Initial Distribution   & $A$   & Difference between $\pi_-$ and state probability after $24$h' \
               ' \\\\\hline\n'

# P   & $K = 0.3 \cdot P$   & $K = 0.5 \cdot P$   &  &  \\ \hline
# \end{tabular}
# \end{table}
qsd_minus = rgen.return_re_model(reverse=False).calculate_qsd()
Q = rgen.return_re_model(full=True).q_matrix[:-1, :-1]

def calculate_distribution(d, t):
    return np.matmul(d, sclig.expm(t * Q))

initial_values = [5, 10, 20, 40, 47]
# initial_values = [20]
for i in initial_values:

    A = rgen.return_re_model(pop=P, K=0.4 * P, initial_dist=i).calculate_bound(probability_epsilon_bound=1e-10)[0]

    dif_minus = np.zeros(len(Q))
    d_minus = np.zeros(len(Q))
    d_minus[i] = 1
    d_minus = calculate_distribution(d_minus, 24)

    for j in range(len(Q)):
        try:
            dif_minus[j] = abs(qsd_minus[j] - d_minus[j])
        except IndexError:
            dif_minus[j] = d_minus[j]

    S = sum(dif_minus)

    latex_line +=  (f'$H={i}$   & ${ure_round_to_significant(A, 2)}$   '
                    f'& ${ure_round_to_significant(S, 2)}$  \\\\\hline\n')

print()
latex_line += '\end{tabular} \n\end{table}'
print(latex_line)