import matplotlib.pyplot as plt
from rate_functions import *
from math import ceil

def search_thr(H):
    """
        This function finds the threshold using dichotomy search.
        It performs a search over sigma in [0, 100] and if dH/dt has three or one equilibrium points
    """

    e_start = 0
    e_finish = 100
    while abs(e_finish - e_start) > 1e-10:
        e = (e_start + e_finish)/2

        reset_l = [-x for x in reset_rate(H)]
        data = sum_l(sum_l(amp_rate(H), reset_l), set_rate(H, e))

        flag_1 = False
        flag_2 = False
        flag_3 = False
        flag_4 = False
        for d in data:

            if d >= 0:
                flag_1 = True
            if d < 0 and flag_1:
                flag_2 = True
            if d >= 0 and flag_2:
                flag_3 = True
            if d < 0 and flag_3:
                flag_4 = True

        assert flag_1
        assert flag_2

        # Case with three roots
        if flag_4:
            e_start = e
        else:
            e_finish = e

    return e

if __name__ == '__main__':
    # 1.8 
    thr = search_thr(np.linspace(start=0, stop=P, num=1000000))
    print(thr)








