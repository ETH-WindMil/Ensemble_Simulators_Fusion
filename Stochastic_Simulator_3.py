'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
plt.style.use(['seaborn-darkgrid'])
import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az
import seaborn as sns
from scipy.stats import weibull_min

plt.style.use('ggplot')

# data = np.random.normal(-5, 10, 50)
# print(data.shape)

n = 100     # number of samples
k = 2.4     # shape
lam = 5.5   # scale
loc = 0     # location parameter
x = weibull_min.rvs(k, loc, lam, n)

y = x**2 + np.sqrt(x) + 100

# Adding noise to the datasets
y += np.random.normal(0, 10, n)
print(y.shape)

plt.figure()
sns.distplot(y, label='Data observations')
plt.show()

print(y.mean())
print(y.std())


# Visualize / plot the datasets
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Assume these are the observations, and we have stochastic simulators
# emulating the observations
data = y


# d = pd.read_csv('./Data/milk.csv')
# d.iloc[:,1:] = d.iloc[:,1:] - d.iloc[:,1:].mean()
# print(d['neocortex'].shape)


# # Model
# with pm.Model() as model_obs:
#     yobs = pm.Normal('yobs', mu=0, sigma=10)
#     trace_obs = pm.sample(100, nuts_kwargs={'target_accept': 0.9}, tune=1000, chains = 4)
#
# print(trace_obs)
# print(pm.summary(trace_obs))
# print(trace_obs.stat_names)
#
# # Trace of MCMC sampler
# pm.traceplot(trace_obs);
# plt.show()
#
# # Plot of the posterior distribution
# pm.plot_posterior(trace_obs);
# plt.show()
#
# print(len(trace_obs['yobs']))
#
# plt.figure()
# sns.distplot(trace_obs['yobs']);
# plt.show()
#
# post_pred_obs = pm.sample_posterior_predictive(trace_obs, samples=5000)
# fig, ax = plt.subplots()
# sns.distplot(post_pred_obs['yobs'].mean(axis=1), label='Posterior predictive means', ax=ax)
# ax.axvline(data.mean(), ls='--', color='r', label='True mean')
# ax.legend();

# # the histogram of the target observed data
# n, bins, patches = plt.hist(trace_obs, 5, density=False, facecolor='g', alpha=0.75)
# plt.xlabel('yobs')
# plt.ylabel('Probability')
# plt.title('Histogram of yobs')
# plt.grid(True)
# plt.show()

#--------------------------------
# Model 0
#--------------------------------
with pm.Model() as model_0:
    y = pm.Normal('y', mu=0.25, sigma=10)
    trace_y0 = pm.sample(1000, nuts_kwargs={'target_accept': 0.9}, tune=1000, chains = 4)

print(trace_y0)
print(pm.summary(trace_y0))

#--------------------------------
# Model 1
#--------------------------------
with pm.Model() as model_1:
    y = pm.Normal('y', mu=-0.5, sigma=9)
    trace_y1 = pm.sample(1000, nuts_kwargs={'target_accept': 0.9}, tune=1000, chains = 4)

print(trace_y1)
print(pm.summary(trace_y1))

#--------------------------------
# Model 2
#--------------------------------
with pm.Model() as model_2:
    y = pm.Normal('y', mu=100, sigma=20)
    trace_y2 = pm.sample(1000, nuts_kwargs={'target_accept': 0.9}, tune=1000, chains = 4)

print(trace_y2)
print(pm.summary(trace_y2))


# Plots for visual comparison of the models
traces = [trace_y0, trace_y1, trace_y2]
pm.densityplot(traces, var_names=['y']);
# pm.forestplot(traces, figsize=(10, 5));
plt.show()

# Pseudo Bayesian model averaging using WAIC
model_dict = dict(zip([model_0, model_1, model_2], traces))
# comp = pm.compare(model_dict, method='stacking')
# comp = pm.compare(model_dict, method='BB-pseudo-BMA')
comp = pm.compare(model_dict, method='pseudo-BMA')
print(comp)
