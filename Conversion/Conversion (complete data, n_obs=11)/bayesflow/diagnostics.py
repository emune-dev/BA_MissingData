import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy.stats import binom
import scipy.stats as stats
from scipy.integrate import quad
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error
#from matplotlib.ticker import FormatStrFormatter

from bayesflow.computational_utilities import expected_calibration_error


def true_vs_estimated(theta_true, theta_est, param_names, figsize=(8, 4), show=True, filename=None, font_size=12):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values. """

    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_est[:, j], theta_true[:, j], color='black', alpha=0.4)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (theta_est[:, j] - theta_true[:, j])**2 ))
        nrmse = rmse / (theta_true[:, j].max() - theta_true[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=12)
        
        # Compute R2
        r2 = r2_score(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=12)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
    
    # Adjust spaces
    f.tight_layout()
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_metrics.png".format(filename), dpi=600, bbox_inches='tight')
    return f


def plot_sbc(theta_samples, theta_test, param_names, bins=25, figsize=(8, 4), interval=0.99, show=True, filename=None, font_size=12):
    """ Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018). """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    N = int(theta_test.shape[0])

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Compute ranks (using broadcasting)    
    ranks = np.sum(theta_samples < theta_test[:, np.newaxis, :], axis=1)
    
    # Compute interval
    endpoints = binom.interval(interval, N, 1 / (bins))

    # Plot histograms
    for j in range(len(param_names)):
        
        # Add interval
        axarr[j].axhspan(endpoints[0], endpoints[1], facecolor='gray', alpha=0.3)
        axarr[j].axhline(np.mean(endpoints), color='gray', zorder=0, alpha=0.5)
        
        sns.histplot(ranks[:, j], kde=False, ax=axarr[j], color='#a34f4f', bins=bins, alpha=0.95)
        
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
        if j == 0:
            axarr[j].set_xlabel('Rank statistic')
        axarr[j].get_yaxis().set_ticks([])
        axarr[j].set_ylabel('')
    
    f.tight_layout()
    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_SBC.png".format(filename), dpi=600, bbox_inches='tight')
    return f
      
        
def plot_posterior(param_samples, posterior_xy, bounds=None, levels=None, filename=None, font_size=12):
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.size'] = font_size
    
    plt.subplot(1, 3, 1)
    # Level sets of analytic posterior distribution
    grid = 201
    A = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, grid)
    B = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, grid)
    true_posterior = np.zeros((grid, grid))
    for iy in range(0, grid):
        for ix in range(0, grid):
            true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])
    if levels is not None:
        true_posterior = plt.contour(A, B, true_posterior, levels, colors='blue')
    else:
        true_posterior = plt.contour(A, B, true_posterior, colors='blue')
        plt.clabel(true_posterior, fontsize=9, inline=1)
    h1, _ = true_posterior.legend_elements()
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], fontsize=11.5)
    plt.xlabel('Parameter $k_1$', fontsize=14)
    plt.ylabel('Parameter $k_2$', fontsize=14)

    # Check marginal densities
    grid = 201
    A = np.linspace(mean_x - 5 * std_x, mean_x + 5 * std_x, grid)
    B = np.linspace(mean_y - 5 * std_y, mean_y + 5 * std_y, grid)
    if bounds is None:
        bounds = np.array([mean_x - 8 * std_x, mean_x + 8 * std_x, mean_y - 8 * std_y, mean_y + 8 * std_y])
    plt.subplot(1, 3, 2)
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label='BayesFlow')
    marginal_x = np.zeros(grid)
    for i in range(grid):
        x = A[i]
        integrand_y = lambda y: posterior_xy(x, y)
        marginal_x[i] = quad(integrand_y, bounds[2], bounds[3])[0]
    plt.plot(A, marginal_x, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=14)
    plt.xlabel('Parameter $k_1$', fontsize=14)
    plt.legend(fontsize=11.5)
    plt.subplot(1, 3, 3)
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label='BayesFlow')
    marginal_y = np.zeros(grid)
    for j in range(grid):
        y = B[j]
        integrand_x = lambda x: posterior_xy(x, y)
        marginal_y[j] = quad(integrand_x, bounds[0], bounds[1])[0]
    plt.plot(B, marginal_y, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=14)
    plt.xlabel('Parameter $k_2$', fontsize=14)
    plt.legend(fontsize=11.5)

    plt.tight_layout()
    plt.show()
    # Save if specified
    if filename is not None:
        fig.savefig("figures/{}_posterior.png".format(filename), dpi=600, bbox_inches='tight')
        
        
def plot_posterior_uniform(param_samples, posterior_xy, bounds=None, levels=None, filename=None, font_size=12):
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.size'] = font_size
    
    plt.subplot(1, 3, 1)
    # Level sets of analytic posterior distribution
    grid = 201
    A = np.linspace(np.min(param_samples[:,0])-0.003, np.max(param_samples[:,0]+0.003), grid)
    B = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, grid)
    true_posterior = np.zeros((grid, grid))
    for iy in range(0, grid):
        for ix in range(0, grid):
            true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])
    if levels is not None:
        true_posterior = plt.contour(A, B, true_posterior, levels, colors='blue')
    else:
        true_posterior = plt.contour(A, B, true_posterior, colors='blue')
        plt.clabel(true_posterior, fontsize=9, inline=1)
    h1, _ = true_posterior.legend_elements()
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], fontsize=11.5)
    plt.xlabel('Parameter $k_1$', fontsize=14)
    plt.ylabel('Parameter $k_2$', fontsize=14)

    # Check marginal densities
    grid = 201
    A = np.linspace(mean_x - 5 * std_x, mean_x + 5 * std_x, grid)
    B = np.linspace(mean_y - 5 * std_y, mean_y + 5 * std_y, grid)
    if bounds is None:
        bounds = np.array([mean_x - 8 * std_x, mean_x + 8 * std_x, mean_y - 8 * std_y, mean_y + 8 * std_y])
    plt.subplot(1, 3, 2)
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label='BayesFlow')
    marginal_x = np.zeros(grid)
    for i in range(grid):
        x = A[i]
        integrand_y = lambda y: posterior_xy(x, y)
        marginal_x[i] = quad(integrand_y, bounds[2], bounds[3])[0]
    plt.plot(A, marginal_x, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=14)
    plt.xlabel('Parameter $k_1$', fontsize=14)
    plt.legend(fontsize=11.5)
    plt.subplot(1, 3, 3)
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label='BayesFlow')
    marginal_y = np.zeros(grid)
    for j in range(grid):
        y = B[j]
        integrand_x = lambda x: posterior_xy(x, y)
        marginal_y[j] = quad(integrand_x, bounds[0], bounds[1])[0]
    plt.plot(B, marginal_y, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=14)
    plt.xlabel('Parameter $k_2$', fontsize=14)
    plt.legend(fontsize=11.5)

    plt.tight_layout()
    plt.show()
    # Save if specified
    if filename is not None:
        fig.savefig("figures/{}_posterior.png".format(filename), dpi=600, bbox_inches='tight')
    
    
def plot_posterior_comparison(param_samples, A, B, C, D, true_posterior, marginal_x, marginal_y, row, method):     
    plt.subplot(4, 3, 3*row-2)    
    levels = np.array([30, 75, 130, 188, 240])
    true_posterior = plt.contour(A, B, true_posterior, levels, colors='blue')
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.title(method, fontsize=23, loc='left', pad=8)
    plt.legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], fontsize=12.5)
    plt.xlabel('Parameter $k_1$', fontsize=16.5)
    plt.ylabel('Parameter $k_2$', fontsize=16.5)

    # Check marginal densities    
    plt.subplot(4, 3, 3*row-1)
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label='BayesFlow')    
    plt.plot(C, marginal_x, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=16.5)
    plt.xlabel('Parameter $k_1$', fontsize=16.5)
    plt.legend(fontsize=11.45)
    
    plt.subplot(4, 3, 3*row)
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label='BayesFlow')    
    plt.plot(D, marginal_y, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=16.5)
    plt.xlabel('Parameter $k_2$', fontsize=16.5)
    plt.legend(fontsize=11.45)
    

def error_metrics(true_params, param_samples):
    v = np.empty(5)  
    param_means = param_samples.mean(axis=1)
    for j in range(2):
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (param_means[:, j] - true_params[:, j])**2 ))
        nrmse = rmse / (true_params[:, j].max() - true_params[:, j].min())
        v[2*j] = nrmse        
        # Compute R²
        r2 = r2_score(true_params[:, j], param_means[:, j])
        v[2*j+1] = r2
    
    # Compute other MSE (as proposed by Yannik)
    n_true = true_params.shape[0]
    n_samples = param_samples.shape[1]
    a = []
    for m in range(n_true):
        mse = mean_squared_error(np.ones((n_samples,1))@np.array([true_params[m]]), param_samples[m], squared=False)
        a.append(mse)  
    v[4] = np.mean(np.array(a))    
    return v