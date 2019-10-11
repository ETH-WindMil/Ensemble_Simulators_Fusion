# https://statmodeling.stat.columbia.edu/2017/05/31/compare-stan-pymc3-edward-hello-world/
# https://towardsdatascience.com/hands-on-bayesian-statistics-with-python-pymc3-arviz-499db9a59501
# Plot individual distributions of posrtiors

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

# Initialize random number generator
SEED = 123
np.random.seed(SEED)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
n = 100

# Predictor variable
X1 = np.random.randn(n)
X2 = np.random.randn(n) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(n)*sigma
print(Y.shape)


with pm.Model() as model_0:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

    # MCMC sample the posterior distributions of the model parameters
    trace_0 = pm.sample(100, nuts_kwargs={'target_accept': 0.9}, tune=1000, chains = 4)

# Detailed summary of the posrterior
print(pm.summary(trace_0))

# https://ericmjl.github.io/bayesian-stats-talk/
# Plot trace of parameters
pm.plot_trace(trace_0);
plt.show()

pm.plot_posterior(trace_0, color='#87ceeb');
plt.show()

# Plot joint-distribution of parameters
pm.plot_joint(trace_0, kind='kde', fill_last=False);
plt.show()
