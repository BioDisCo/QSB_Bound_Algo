import numpy as np


######################################################
# Parameters
K = 50
# TODO write V
# K = 8e7-8e8 (or even lower values are possible) cells/mL
P = 120
# P = 1e8-1e9 cells/mL

n = 4

# Rates before rescaling
k = 35 # h-1
p = 14 # h-1

# Rates after rescaling - consider 8760 hours in an year

# sigma = 0
sigma = 0.5
# sigma_tr = 1.8 # h-1
sgvl = [0, 1.5, 3.04, 4, 5]
normalization_factor = 10
Hl = np.linspace(start=0, stop=P, num=P)
######################################################

#####################################################
# Font and others
title_font = 15
axis_font = 12
#####################################################