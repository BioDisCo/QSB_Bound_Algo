import matplotlib.pyplot as plt
from rate_functions import *

b_d = sum_l(set_rate(Hl), amp_rate(Hl))
b_d = sum_l(b_d, neg(reset_rate(Hl)))

plt.plot(Hl, b_d)
plt.plot(Hl, zero_line(Hl))
plt.show()


