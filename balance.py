import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import itertools

def func(x, a,b):
    return x*a + b

CLR="rgb"

def fit_teams(teams):
    xdata = [np.arange(len(t),dtype=float) for t in teams]
    ydata = [np.array(t,dtype=float) for t in teams]
    for y in ydata:
        y.sort()

    for i in range(len(teams)):
        popt, pcov = curve_fit(func, xdata[i], ydata[i])
        print(pcov)
        print(np.linalg.cond(pcov))
        plt.plot(xdata[i], ydata[i], "*", color=CLR[i])
        plt.plot(xdata[i], func(xdata[i], *popt), color=CLR[i],label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

players = [1500, 1700, 1900, 2200, 1600,1800,2100,2150]

combs = itertools.combinations(players,4)
print(list(combs))
fit_teams([[1500, 1700, 1900, 2200], [1600,1800,2100,2150]])
plt.legend()
plt.show()
