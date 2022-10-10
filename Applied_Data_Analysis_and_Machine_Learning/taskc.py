from ensurepip import bootstrap
from statistics import variance
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.pipeline import make_pipeline
from numpy.core import _asarray
from sklearn.utils import resample
import time
from utils import ( 
    FrankeFunction, generate_determ_data, create_X, create_simple_X,
    compute_optimal_parameters, compute_optimal_parameters_inv, generate_design_matrix, predict, MSE)

def plot_OLS_boot_figs(MSE_train, MSE_test, var, bias, degs):
    
    fig, axs = plt.subplots(1,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0].plot(degs, MSE_train, 'b', label='MSE_train') 
    axs[0].plot(degs, MSE_test, 'r', label='MSE_test')
    axs[0].set_xlabel('Polynomial order')
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].legend()
    
    axs[1].plot(degs, MSE_test, 'b', label='MSE_test') 
    axs[1].plot(degs, var, 'g', label='variance')
    axs[1].plot(degs, bias, 'y', label='bias')
    axs[1].set_xlabel('Polynomial order')
    axs[1].set_ylabel('R2 Score')
    axs[1].legend()

    plt.show() 

def OLS_reg_boot(x, y,z=None, n_points=20, degrees=5, n_boots=10, scaling=False, noisy=True, r_seed=427): 
    np.random.seed(r_seed)
    if z == None: 
        z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    bias = np.zeros(degrees)
    variance = np.zeros(degrees)
    polydegree = np.zeros(degrees)
    
    for degree in range(1, degrees+1): 
        
        print(f'Processing polynomial of {degree} degree ')
        x_train, x_test, z_train, z_test = train_test_split(X[:, :int((degree+1)*(degree+2)/2)], z.ravel(), test_size=0.2)
        if scaling: 
            x_train_mean = np.mean(x_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)  
            x_train -= x_train_mean
            x_test -= x_train_mean
            z_train_centered = z_train - z_train_mean
        else: 
            z_train_mean = 0
            z_train_centered = z_train

        pred_train_avg = np.empty((n_boots, z_train.shape[0]))
        pred_test_avg = np.empty((n_boots, z_test.shape[0]))

        for j in range(n_boots): 
            
            # Bootstrap resampling of datasets after split
            X_, z_ = resample(x_train, z_train_centered, replace=True)
            beta_SVD = compute_optimal_parameters(X_, z_)
        
            z_pred_train = predict(x_train, beta_SVD, z_train_mean) 
            z_pred_test = predict(x_test, beta_SVD, z_train_mean)

            pred_train_avg[j, :] = z_pred_train
            pred_test_avg[j, : ] = z_pred_test
        
        MSE_train_list[degree-1] = MSE(z_train, pred_train_avg)#np.mean(np.mean((pred_train_avg-z_train)**2, axis=0, keepdims=True))
        MSE_test_list[degree-1] = MSE(z_test, pred_test_avg) #np.mean(np.mean((pred_test_avg-z_test)**2, axis=0, keepdims=True))
        polydegree[degree-1] = degree
        bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))

    return bias, variance, MSE_train_list, MSE_test_list, polydegree

def task_c(n_points=20, noisy=True, centering=True,  degrees=11, n_boots=100, r_seed=79):

    x, y = generate_determ_data(n_points)
        
    #bias,var, MSE_train, MSE_test, pol = OLS_boot_reg(n_points=20, degrees=11, n_boots=100, r_seed=79, scaling=True)
    bias,var, MSE_train, MSE_test, pol = OLS_reg_boot(x,y,n_points=n_points, degrees=degrees, n_boots=n_boots, r_seed=r_seed, scaling=centering)
    plot_OLS_boot_figs(MSE_train, MSE_test, var, bias, pol)

    # Random seeds list when using create_X
    # Good random seeds = [2, 4, 5, 9, 14, 17, 79
    # Good, but some random behavoiur of data sets = [7, 8, 15 
    # Medium random seeds = [10, 12, 1911, 16, 19, 20
    # Weak random seeds = [6, 11, 31, 18

task_c()