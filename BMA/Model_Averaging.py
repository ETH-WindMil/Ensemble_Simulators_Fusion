'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
plt.style.use(['seaborn-darkgrid'])
import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az

d = pd.read_csv('../Data/milk.csv')
d.iloc[:,1:] = d.iloc[:,1:] - d.iloc[:,1:].mean()
d.head()
print(d)
print(d.shape)

# the histogram of the target observed data
n, bins, patches = plt.hist(d['kcal.per.g'], 5, density=False, facecolor='g', alpha=0.75)
plt.xlabel('kcal.per.g')
plt.ylabel('Probability')
plt.title('Histogram of kcal.per.g')
plt.grid(True)
plt.show()

# Three competing models describing the same physical phenomena
# Model 0
with pm.Model() as model_0:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta  = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', 10)

    mu = alpha + beta * d['neocortex']

    kcal = pm.Normal('kcal', mu=mu, sigma=sigma, observed=d['kcal.per.g'])
    trace_0 = pm.sample(1000, nuts_kwargs={'target_accept': 0.9}, tune=1000)

# Model 1
with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', 10)

    mu = alpha + beta * d['log_mass']

    kcal = pm.Normal('kcal', mu=mu, sigma=sigma, observed=d['kcal.per.g'])
    trace_1 = pm.sample(1000, nuts_kwargs={'target_accept': 0.9}, tune=1000)

# Model 2
with pm.Model() as model_2:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=2)
    sigma = pm.HalfNormal('sigma', 10)

    mu = alpha + pm.math.dot(beta, d[['neocortex','log_mass']].T)

    kcal = pm.Normal('kcal', mu=mu, sigma=sigma, observed=d['kcal.per.g'])
    # trace_2 = pm.sample(10000, nuts_kwargs={'target_accept': 0.9}, tune=1000)
    trace_2 = pm.sample(1000, nuts_kwargs={'target_accept': 0.9}, tune=1000)

print(trace_0)

# Plots for visual comparison of the models
traces = [trace_0, trace_1, trace_2]
pm.densityplot(traces, var_names=['alpha', 'sigma']);
# pm.forestplot(traces, figsize=(10, 5));
plt.show()

# Model averaging 
model_dict = dict(zip([model_0, model_1, model_2], traces))
comp = pm.compare(model_dict, method='stacking')
# comp = pm.compare(model_dict, method='BB-pseudo-BMA')
# comp = pm.compare(model_dict, method='pseudo-BMA')
print(comp)
# az.compare({'model_0':trace_0, 'model_1':trace_1, 'model_2':trace_2}, method='BB-pseudo-BMA')

# Now we are going to use the previously copmuted weights to generate predictions based not
# on a single model but on the weighted set of models. This is one way to perform
# model averaging. Using PyMC3 we can call the sample_posterior_predictive_w
# function as follows:
ppc_w = pm.sample_posterior_predictive_w(traces, 1000, [model_0, model_1, model_2],
                        weights=comp.weight.sort_index(ascending=True),
                        progressbar=False)

# We are also going to compute PPCs for the lowest-WAIC model
ppc_2 = pm.sample_posterior_predictive(trace_2, 1000, model_2,
                     progressbar=False)

# A simple way to compare both kind of predictions is to plot their mean and hpd interval
mean_w = ppc_w['kcal'].mean()
hpd_w = pm.hpd(ppc_w['kcal']).mean(0)
mean = ppc_2['kcal'].mean()
hpd = pm.hpd(ppc_2['kcal']).mean(0)

plt.errorbar(mean, 1, xerr=[[mean-hpd[0]] , [hpd[1]-mean]], fmt='o', label='model 2')
plt.errorbar(mean_w, 0, xerr=[[mean_w-hpd_w[0]] , [hpd_w[1]-mean_w]], fmt='o', label='weighted models')
plt.yticks([])
plt.ylim(-1, 2)
plt.xlabel('kcal per g')
plt.legend();
plt.show()
