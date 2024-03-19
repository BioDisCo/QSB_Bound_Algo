import scipy.linalg as sclig
import numpy as np

def round_to_significant(number, decimals, base_number=0):

    power = 0
    while number < 1*10**base_number:
        power = power + 1
        number = number*10

    number = round(number, decimals)
    if power > 0:
        return str(number) + '\cdot10^{-' + str(power) + '}'
    else:
        return str(number)


def calculate_distribution(d, t, Q):
    return np.matmul(d, sclig.expm(t * Q))

