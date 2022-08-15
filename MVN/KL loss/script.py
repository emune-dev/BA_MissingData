#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal


# ## Simulator settings

# In[4]:

mu_0 = np.zeros(2)   # prior mean
sigma_0 = np.identity(2)   # prior covariance matrix
cov = np.array([[0.5,-0.35],[-0.35,1]])   # covariance matrix of likelihood


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim)
    """
    # Prior distribution: mu ~ N(mu_0, sigma_0)
    p_samples = np.random.multivariate_normal(mu_0, sigma_0, batch_size)
    return p_samples.astype(np.float32)


def batch_simulator(prior_samples, n_obs):   # n_obs (number of observations in each dataset)
    """
    Simulate multiple MVN model datasets
    """
    n_sim = prior_samples.shape[0]   # batch size
    sim_data = np.empty((n_sim, n_obs, 2), dtype=np.float32)   # 1 batch consisting of n_sim datasets, each with n_obs observations

    for m in range(n_sim):
        mean = prior_samples[m]
        sim_data[m] = np.random.multivariate_normal(mean, cov, n_obs)

    return sim_data   


# Fixed n_obs
n_obs = 1


# In[9]:

# Monte Carlo estimate for correction constant in KL loss
b = np.zeros(30)

for k in range(30):
    n_MC = 100000
    prior_params = prior(n_MC)
    x_data = batch_simulator(prior_params, n_obs)
    constant = np.zeros(n_MC)
    for i in range(n_MC):
        B = inv(sigma_0 + cov)
        m = B @ x_data[i, 0]
        Lambda = B @ cov
        posterior_eval = lambda theta: multivariate_normal.pdf(theta, m, Lambda)
        constant[i] = np.log(posterior_eval(prior_params[i]))
    c = np.mean(constant) + np.log(2 * np.pi)
    b[k] = c

print(b)
print(np.mean(b))
