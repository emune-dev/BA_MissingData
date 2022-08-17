#!/usr/bin/env python
# coding: utf-8

# # Posterior Estimation - MVN model ($D=2, n_{obs}=1$)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from numpy.linalg import inv
from scipy.stats import multivariate_normal

from bayesflow.networks import InvertibleNetwork, FlattenNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel



# ## Simulator settings

# In[3]:


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


# We build an amortized parameter estimation network.

# In[4]:


bf_meta = {
    'n_coupling_layers': 3,
    's_args': {
        'units': [32, 32, 32],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    't_args': {
        'units': [32, 32, 32],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    'n_params': 2
}


# In[5]:


summary_net = FlattenNetwork()
inference_net = InvertibleNetwork(bf_meta)
amortizer = SingleModelAmortizer(inference_net, summary_net)


# We connect the prior and simulator through a *GenerativeModel* class which will take care of forward inference.

# In[6]:


generative_model = GenerativeModel(prior, batch_simulator)


# In[7]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True,
)


# In[8]:


trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model,
    learning_rate = lr_schedule,
    checkpoint_path = './MVN1_3ACB_[32,32,32]_slow_ckpts',
    max_to_keep=75,
    skip_checks=True
)


# ### Online training

# In[9]:


# Fixed n_obs
n_obs = 1


# In[9]:


losses = trainer.train_online(epochs=75, iterations_per_epoch=1000, batch_size=128, n_obs=n_obs)
print(losses)
