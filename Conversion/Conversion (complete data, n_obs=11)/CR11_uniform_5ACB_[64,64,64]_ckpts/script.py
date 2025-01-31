#!/usr/bin/env python
# coding: utf-8

# # Parameter Estimation Workflow for conversion reaction

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel



# ## Simulator settings

# In[3]:


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) 
    """    
    # Prior range for log-parameters: k_1, k_2 ~ U(-1.5, 0)
    p_samples = np.random.uniform(low=(-1.5, -1.5), high=(0., 0.), size=(batch_size, 2))
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


# We build an amortized parameter estimation network.

# In[4]:


bf_meta = {
    'n_coupling_layers': 5,
    's_args': {
        'units': [64, 64, 64],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    't_args': {
        'units': [64, 64, 64],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    'n_params': 2
}


# In[5]:


summary_net = None
inference_net = InvertibleNetwork(bf_meta)
amortizer = SingleModelAmortizer(inference_net, summary_net)


# We connect the prior and simulator through a *GenerativeModel* class which will take care of forward inference.

# In[6]:


generative_model = GenerativeModel(prior, batch_simulator)


# In[7]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.95,
    staircase=True,
)


# In[8]:


trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model,
    learning_rate = lr_schedule,
    checkpoint_path = './CR11_uniform_5ACB_[64,64,64]_ckpts',
    max_to_keep=300,
    skip_checks=True
)


# ### Online training

# In[9]:


losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=128, n_obs=n_obs)
print(losses)
