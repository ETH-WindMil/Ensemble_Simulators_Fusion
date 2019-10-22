import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import math as m
from scipy.stats import weibull_min


# Always good to set a seed for reproducibility
SEED = 222
np.random.seed(SEED)

def y_target(x):
    """ returns the target function """
    return ((6*x - 2)**2) * np.sin(12*x - 4) + 12

n = 500    # number of samples
# k = 2.4     # shape
# lam = 5.5   # scale
# loc = 0     # location parameter
# x = weibull_min.rvs(k, loc, lam, n)
x = np.random.uniform(0,1,n)
print(x.shape)

"""
target
"""
xt = np.linspace(0, 1, num=n)
y_t = y_target(xt)
COV = 0.00
y_t += np.random.normal(0, y_t*COV)


"""
Simulator 1
"""
y_1 = y_target(x) + 0
COV = 0.05
y_1 += np.random.normal(0, y_1*COV)

"""
Simulator 2
"""
y_2 = y_target(x) + 1
COV = 0.05
y_2 += np.random.normal(0, y_2*COV)

"""
Simulator 3
"""
A3 = 0.5
B3 = 8
C3 = 2
y_3 = A3*y_target(x**2) + B3*(x**3-0.5)+C3
COV = 0.10
y_3 += np.random.normal(0, y_3*COV)

"""
Simulator 4
"""
A4 = 0.5
B4 = 10
C4 = 5
y_4 = A4*y_target(x) + B4*(x-0.5)+C4
COV = 0.10
y_4 += np.random.normal(0, y_4*COV)

"""
Simulator 5
"""
A5 = 0.25
B5 = 0.5
C5 = 12
y_5 = A5*y_target(np.sqrt(x) + B5) + C5
COV = 0.10
y_5 += np.random.normal(0, y_5*COV)




"""
Visualize / plot the datasets
"""
g1 = (x,y_1)
g2 = (x,y_2)
g3 = (x,y_3)
g4 = (x,y_4)
g5 = (x,y_5)
gt = (xt,y_t)
data = (g1, g2, g3, g4, g5, gt)
colors = ("r", "g", "b", "m", "b","k")
fillers = ('r', 'none', 'none', 'none', 'none', 'k')
groups = ("Simulator 1", "Simulator 2", "Simulator 3", "Simulator 4", "Simulator 5", "Target Response")
mrkrs = ('+', '*', '<', 'd', '.', 'o')
msize = (20, 20, 20, 20, 20, 40)

fig, ax = plt.subplots(figsize=(10, 8))
for data, color, group, mrkr, sze, flrs in zip(data, colors, groups, mrkrs, msize, fillers):
    x_sim, y_sim = data
    ax.scatter(x_sim, y_sim, alpha=0.9, c=flrs, s=sze, marker=mrkr, label=group, edgecolors=color)
ax.legend(loc=2, prop={'size': 18})
ax.grid(True, alpha=0.3, lw=2)
plt.xlabel('X', fontsize=20)
plt.ylabel('Y', fontsize=20)
plt.xlim(xmax = 1, xmin = 0)
plt.ylim(ymax = 35, ymin = 0)
plt.show()

print(data)

"""
Compare distributions of simulators stochastic output
"""
fig, ax = plt.subplots(figsize=(10, 8))
sns.distplot(y_1, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3, 'linestyle':'--'},
                  label = groups[0])
sns.distplot(y_2, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3, 'linestyle':':'},
                  label = groups[1])
sns.distplot(y_3, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3},
                  label = groups[2])
sns.distplot(y_4, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3},
                  label = groups[3])
sns.distplot(y_5, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3, 'linestyle':'-.'},
                  label = groups[4])
sns.distplot(y_t, hist = False, kde = True,
                 kde_kws = {'shade': False, 'linewidth': 3},
                  label = groups[5])
plt.xlim(xmax = 35, xmin = 0)
plt.legend(loc=1, prop={'size': 18})
plt.xlabel('Y', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.title('Kernel Density Estimates of the responses', fontsize=18)
plt.show()
