from parameters import *
import matplotlib.pyplot as plt

steady_state_ratios = []

for i in range(P):

    rr = (i + 1)*p
    er = sigma*(P - i)
    hr = (P - i)*k*i**n/(i**n + K**n)
    steady_state_ratios.append(rr/(er + hr))

# Final ratio is final steady state vector over itself
steady_state_ratios.append(1)

ratios_from_qp = [1]
for i in range(1, P):
    ratios_from_qp.append(ratios_from_qp[i - 1]*steady_state_ratios[-i])

state_vector = [1/sum(ratios_from_qp)]

for i in range(1, P):
    state_vector.append(state_vector[i - 1]*steady_state_ratios[-i])

state_vector.reverse()

plt.plot(state_vector)
plt.title(r"Steady State Distribution")
plt.xlabel("H (count)")
plt.ylabel("State Probability")
plt.tight_layout()

plt.savefig("figures/ssd_figure.png")

plt.show()




