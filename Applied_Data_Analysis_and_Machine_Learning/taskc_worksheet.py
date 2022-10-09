from ensurepip import bootstrap
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

def OLS_boot_reg(n_points=20, degrees=5, n_boots=10, scaling=False, noisy=True, r_seed=427): 
    np.random.seed(r_seed)
    x, y = generate_determ_data(n_points)
    z = FrankeFunction(x,y,noise=noisy)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)

    bias = np.zeros(degrees)
    variance = np.zeros(degrees)
    polydegree = np.zeros(degrees)
    X = create_X(x,y,degrees, centering=scaling)
    i, i2 = 3, 3

    for degree in range(1, degrees+1): 

        print(f'Polynomial degree {degree}')

        if scaling: 
            x_train, x_test, z_train, z_test = train_test_split(X[:, :i], z.ravel(), test_size=0.2)
            x_train_mean = np.mean(x_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)     

            # Using centered values of X and y to compute parameters beta
            x_train_centered = x_train - x_train_mean
            z_train_centered = z_train - z_train_mean
            x_test_centered = x_test - x_train_mean 

            pred_train_avg = np.empty((n_boots, z_train.shape[0]))
            pred_test_avg = np.empty((n_boots, z_test.shape[0]))

            for j in range(n_boots): 
            
                # Bootstrap resampling of datasets after split
                x_, z_ = resample(x_train_centered, z_train_centered, replace=True)
                beta_SVD = compute_optimal_parameters(x_, z_)

                z_pred_train = predict(x_train_centered, beta_SVD, z_train_mean) 
                z_pred_test = predict(x_test_centered, beta_SVD, z_train_mean)

                pred_train_avg[j, :] = z_pred_train
                pred_test_avg[j, : ] = z_pred_test
            i += i2
            i2 += 1 
            
        else: 
            x_train, x_test, z_train, z_test = train_test_split(X[:, :i], z.ravel(), test_size=0.2)

            pred_train_avg = np.empty((n_boots, z_train.shape[0]))
            pred_test_avg = np.empty((n_boots, z_test.shape[0]))

            for j in range(n_boots): 
                
                # Bootstrap resampling of datasets after split
                x_, z_ = resample(x_train, z_train, replace=True)
                beta_SVD = compute_optimal_parameters(x_, z_)

                z_pred_train = predict(x_train, beta_SVD) 
                z_pred_test = predict(x_test, beta_SVD)

                pred_train_avg[j, :] = z_pred_train
                pred_test_avg[j, : ] = z_pred_test
            i += i2
            i2 += 1 
        
        MSE_train_list[degree-1] = np.mean(np.mean((pred_train_avg-z_train)**2, axis=0, keepdims=True))#training_error/n_boots
        MSE_test_list[degree-1] = np.mean(np.mean((pred_test_avg-z_test)**2, axis=0, keepdims=True))#test_error/n_boots
        polydegree[degree-1] = degree
        bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))
       
    return bias, variance, MSE_train_list, MSE_test_list, polydegree

def plot_OLS_boot_figs(*args):
    
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0,0].plot(args[4], args[0], 'b', label='MSE_train') 
    axs[0,0].plot(args[4], args[1], 'r', label='MSE_test')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Mean Squared Error')
    axs[0,0].legend()
    
    axs[0,1].plot(args[4], args[1], 'b', label='MSE_test') 
    axs[0,1].plot(args[4], args[2], 'g', label='variance')
    axs[0,1].plot(args[4], args[3], 'y', label='bias')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('R2 Score')
    axs[0,1].legend()

    plt.show() 

#bias,var, MSE_train, MSE_test = OLS_boot_reg(n_points=40, degrees=11, n_boots=20, seed=9)
bias,var, MSE_train, MSE_test, pol = OLS_boot_reg(n_points=20, degrees=11, n_boots=100, r_seed=79, scaling=True)
plot_OLS_boot_figs(MSE_train, MSE_test, var, bias, pol)

# Random seeds list when using create_X
# Good random seeds = [2, 4, 5, 9, 14, 17, 79, 90210
# Good, but some random behavoiur of data sets = [7, 8, 15 
# Medium random seeds = [10, 12, 1911, 16, 19, 20
# Weak random seeds = [6, 11, 31, 18