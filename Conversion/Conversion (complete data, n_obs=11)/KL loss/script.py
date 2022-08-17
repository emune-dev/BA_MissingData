#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import norm
from scipy.integrate import dblquad


# ## Simulator settings

# In[4]:

def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim)
    """
    # Prior range for log-parameters: k_1, k_2 ~ N(-0.75, 0.25²) iid.
    p_samples = np.random.normal(-0.75, 0.25, size=(batch_size, 2))
    return p_samples.astype(np.float32)


# ODE model for conversion reaction
sigma = 0.015   # noise standard deviation
n_obs = 11
time_points = np.linspace(0, 10, n_obs)


def batch_simulator(prior_samples, n_obs):   # n_obs (number of observations in each dataset)
    """
    Simulate multiple conversion model datasets via analytical solution of ODE
    """
    n_sim = prior_samples.shape[0]   # batch size
    sim_data = np.empty((n_sim, n_obs), dtype=np.float32)   # 1 batch consisting of n_sim datasets, each with n_obs observations

    for m in range(n_sim):
        theta = 10**prior_samples[m]
        s = theta[0] + theta[1]
        b = theta[0]/s
        state_2 = lambda t: b - b * np.exp(-s*t)
        sol = state_2(time_points)
        sim_data[m] = sol + np.random.normal(0, sigma, size = n_obs)   # observable: y = x_2 + N(0,sigma²)

    return sim_data   


# In[9]:

# Monte Carlo estimate for correction constant in KL loss
b = np.zeros(30)

for k in range(30):
    n_MC = 100
    prior_params = prior(n_MC)
    x_data = batch_simulator(prior_params, n_obs)
    constant = np.zeros(n_MC)
    for i in range(n_MC):
        def prior_eval(x, y):
            return norm.pdf(x, -0.75, 0.25) * norm.pdf(y, -0.75, 0.25)

        def likelihood(x, y):
            x = 10 ** x
            y = 10 ** y
            s = x + y
            b = x / s
            state_2 = lambda t: b - b * np.exp(-s * t)
            sol = state_2(time_points)
            residual = (x_data[i] - sol) / sigma
            nllh = np.sum(np.log(2 * np.pi * sigma ** 2) + residual ** 2) / 2
            return np.exp(-nllh)

        def unnormalized_posterior(x, y):
            return likelihood(x, y) * prior_eval(x, y)

        scaling_factor = dblquad(unnormalized_posterior, -2.25, 0.75, lambda y: -2.25, lambda y: 0.75)
        posterior_xy = lambda x, y: unnormalized_posterior(x, y) / scaling_factor[0]
        posterior_eval = lambda theta: posterior_xy(theta[0], theta[1])
        constant[i] = np.log(posterior_eval(prior_params[i]))
    c = np.mean(constant) + np.log(2 * np.pi)
    b[k] = c

print(b)
print(np.mean(b))
