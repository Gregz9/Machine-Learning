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
    FrankeFunction, generate_determ_data, create_X, compute_optimal_parameters, 
    compute_optimal_parameters_inv, generate_design_matrix, predict, MSE, KFold_split,
)

def OLS_cross_reg(n_points=20, degrees=5, k=5, scaling=False, noisy=True, r_seed=9): 
    np.random.seed(r_seed)
    x,y = generate_determ_data(n_points)
    z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees)
    z = z.ravel()
    train_indices, test_indices = KFold_split(z,k)

    print(test_indices[0].shape)

    MSE_train = np.empty(degrees)
    MSE_test = np.empty(degrees)
    bias = np.zeros(degrees)
    variance = np.zeros(degrees)
    polydegree = np.zeros(degrees)
    x_train, z_train = X[train_indices[0], :3], z[train_indices[0]] 
    #print(x_train.shape)
    #x_test, z_test = X[test_indices[0]], z[test_indices[0]]
    
    i, i2 = 3,3
    for degree in range(1, degrees+1): 
        pred_train_avg = np.empty((k, train_indices.shape[1]))
        pred_test_avg = np.empty((k, test_indices.shape[1]))
        training_error = 0 
        test_error = 0
        print(f'Polynomial degree {degree}')
        start_time = time.time()
        for train_indx, test_indx, fold in zip(train_indices, test_indices, [i for i in range(k)]):
            x_train, z_train = X[train_indx, :i], z[train_indx]
            x_test, z_test = X[test_indx, :i], z[test_indx]
            #print(x_train.shape, z_train.shape)

            betas = compute_optimal_parameters(x_train, z_train)
            z_pred_train = predict(x_train, betas)
            z_pred_test = predict(x_test, betas)
            #print(fold)

            pred_train_avg[fold, :] = z_pred_train
            pred_test_avg[fold, :] = z_pred_test 

            training_error += -MSE(z_train, z_pred_train)
            test_error += -MSE(z_test, z_pred_test)
        print("Time used %s seconds" % (time.time() - start_time))

        i += i2 
        i2 += 1
        MSE_train[degree-1] = np.mean(np.mean((pred_train_avg-z_train)**2, axis=0, keepdims=True))#training_error/n_boots
        MSE_test[degree-1] = np.mean(np.mean((pred_test_avg-z_test)**2, axis=0, keepdims=True))#test_error/n_boots
        bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))

        return bias, variance, MSE_train, MSE_test, polydegree
OLS_cross_reg()

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

bias,var, MSE_train, MSE_test, pol = OLS_cross_reg(n_points=20, degrees=10)
plot_OLS_boot_figs(MSE_train, MSE_test, var, bias, pol)