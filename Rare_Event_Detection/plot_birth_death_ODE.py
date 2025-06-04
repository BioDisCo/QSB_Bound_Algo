import matplotlib.pyplot as plt
from rate_functions import *

P = 1600
Hl = np.linspace(start=0, stop=P, num=P)

b_d = sum_l(set_rate(Hl, P=P), amp_rate(Hl, K=0.3*P, P=P))
b_d = sum_l(b_d, neg(reset_rate(Hl)))

plt.plot(Hl, b_d)
plt.plot(Hl, zero_line(Hl))
plt.show()


