# BA_MissingData

This repository contains all code and simulation scripts for my Bachelor thesis "Missing Data Handling in Amortized Bayesian Inference via Invertible Neural Networks".
It is divided into four folders dedicated to the four investigated models: Conversion reaction model, multivariate normal distribution model (MVN), oscillation model and SIR model.

Regarding the content of these folders: 
- The Jupyter notebooks were used to validate/compare the performance of trained BayesFlow workflows and to create illustrative figures for the thesis. 
- Subfolders with the ending "ckpts" contain the Python script for training a BayesFlow workflow on a specific forward model, an output file of the running loss as well as the stored networks after the final training epoch. 
- Subfolders with the name "KL loss" contain the Python script for and results of computing the model-specific correction constant for the KL loss.
- Subfolders with the name "bayesflow" contain the implementation of the BayesFlow method downloaded from https://github.com/stefanradev93/BayesFlow. In some cases, slight modifications have been made to meet the purpose of our numerical experiment.
