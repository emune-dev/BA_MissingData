#!/usr/bin/env python
# coding: utf-8

# # Parameter Estimation - SIR model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from scipy.stats import norm
from scipy.integrate import solve_ivp, dblquad
import random

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
    # Prior range for log-rate parameters: 
    b_samples = np.random.normal(-1.0, 0.3, size=(batch_size, 1))
    c_samples = np.random.normal(-1.5, 0.3, size=(batch_size, 1))
    p_samples = np.c_[b_samples, c_samples]
    return p_samples.astype(np.float32)


# ODE model for SIR dynamics
def sir_dynamics(t, x, theta):
    theta = 10**theta
    return np.array([
            -theta[0]*x[0]*x[1]/N, 
            theta[0]*x[0]*x[1]/N - theta[1]*x[1],
            theta[1]*x[1]
            ])

N = 1000   # population size
x0 = np.array([999, 1, 0])   # initial state       
sigma = 0.015   # noise standard deviation
t_end = 180
n_obs = 31
time_points = np.linspace(0, t_end, n_obs)
missing_max = 15


def batch_simulator(prior_samples, n_obs):  
    """
    Simulate multiple SIR model datasets with missing values and augmentation by zeros/ones
    """    
    n_sim = prior_samples.shape[0]   # batch size    
    sim_data = np.ones((n_sim, n_obs, 4), dtype=np.float32)   # 1 batch consisting of n_sim datasets, each with n_obs observations
    n_missing = np.random.randint(0, missing_max+1, size=n_sim) 
    
    for m in range(n_sim):
        rhs = lambda t,x: sir_dynamics(t, x, prior_samples[m])
        sol = solve_ivp(rhs, t_span=(0, t_end), y0=x0, t_eval=time_points, atol=1e-9, rtol=1e-6)
        sim_data[m, :, 0:3] = sol.y.T/N + np.random.normal(0, sigma, size=(n_obs, 3))     # observable: y = x + N(0,sigma²)
        
        # artificially induce missing data
        missing_indices = random.sample(range(n_obs), n_missing[m])
        sim_data[m][missing_indices] = np.array([-1.0, -1.0, -1.0, 0.0])
        
    return sim_data   


# We build an amortized parameter estimation network.

# In[4]:


bf_meta = {
    'n_coupling_layers': 5,
    's_args': {
        'units': [128, 128, 128],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    't_args': {
        'units': [128, 128, 128],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    'n_params': 2
}


# In[5]:


summary_net = LSTM(128)
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
    checkpoint_path = './SIR_augment01_5ACB_[128,128,128]_LSTM(128)_ckpts',
    max_to_keep=300,
    skip_checks=True
)


# ### Online training

# In[9]:


losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=64, n_obs=n_obs)
print(losses)
