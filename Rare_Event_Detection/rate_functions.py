import numpy as np

from parameters import *

def set_rate(Hl, s=sigma, P=P):
    return [s*(P - H) for H in Hl]

def reset_rate(Hl, r=p):
    return [r*H for H in Hl]

def amp_rate(Hl, k=k, K=K, n=n, P=P):
    return [k*(P - H)*H**n/(H**n + K**n) for H in Hl]

def zero_line(Hl):
    return [0 for _ in Hl]

def sum_l(L1, L2):
    return [e1 + e2 for e1, e2 in zip(L1, L2)]

def neg(L1):
    return [-e for e in L1]

def Q_matrix(Hl, set_f=set_rate, reset_f=reset_rate, amp_f=amp_rate):

    if len(Hl) == 1:
        raise TypeError

    matrix = np.zeros((len(Hl), len(Hl)))

    l_set_rates = set_f(Hl)
    l_reset_rates = reset_f(Hl)
    l_amp_rates = amp_f(Hl)

    for i in range(len(Hl)):

        if i == 0:
            matrix[0, 1] = l_set_rates[i] + l_amp_rates[i]
            matrix[0, 0] = -(l_set_rates[i] + l_amp_rates[i])
        elif i == len(Hl) - 1:
            matrix[i, i - 1] = l_reset_rates[i]
            matrix[i, i] = -l_reset_rates[i]
        else:
            matrix[i, i - 1] = l_reset_rates[i]
            matrix[i, i + 1] = l_set_rates[i] + l_amp_rates[i]
            matrix[i, i] = -(l_reset_rates[i] + l_set_rates[i] + l_amp_rates[i])

    return matrix


def Q_cut(Hl, set_f=set_rate, reset_f=reset_rate, amp_f=amp_rate, stop_index=-1):

    l_set_rates = set_f(Hl)
    l_reset_rates = reset_f(Hl)
    l_amp_rates = amp_f(Hl)

    flag = 0
    final_index = 0
    if stop_index != -1:
        final_index = stop_index
    else:
        for i in range(len(Hl)):

            if l_set_rates[i] + l_amp_rates[i] > l_reset_rates[i] and flag == 0:
                continue

            if l_set_rates[i] + l_amp_rates[i] <= l_reset_rates[i]:
                flag = 1

            if l_set_rates[i] + l_amp_rates[i] > l_reset_rates[i] and flag == 1:
                final_index = i
                break

    matrix = np.zeros((final_index, final_index))

    for i in range(final_index):

        if i == 0:
            matrix[0, 1] = l_set_rates[i] + l_amp_rates[i]
            matrix[0, 0] = -(l_set_rates[i] + l_amp_rates[i])
        elif i == final_index - 1:
            #matrix[i, i - 1] = l_reset_rates[i]
            #matrix[i, i] = -l_reset_rates[i]
            break
        else:
            matrix[i, i - 1] = l_reset_rates[i]
            matrix[i, i + 1] = l_set_rates[i] + l_amp_rates[i]
            matrix[i, i] = -(l_reset_rates[i] + l_set_rates[i] + l_amp_rates[i])

    return matrix


def conditional_Q_cut(Q_cut_m):
    Q = Q_cut_m
    size = Q.shape[0]

    Q[size - 2, -2] = -Q[size - 2, -3]
    Q = Q[:size - 1, :size - 1]
    return Q


def Jump_Chain(Q):
    J = np.zeros(Q.shape)

    for i in range(Q.shape[0]):
        if Q[i, i] == 0:
            continue

        if i < len(Q) - 1:
            J[i, i + 1] = Q[i, i + 1]/-Q[i, i]
        if i > 0:
            if Q[i, i] == 0:
                J[i, i - 1] = 0
            else:
                J[i, i - 1] = Q[i, i - 1]/-Q[i, i]

    return J

if __name__ == '__main__':
    Hl = np.linspace(start=0, stop=P, num=5)
    Q_matrix(Hl, set_rate, reset_rate, amp_rate)







