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
from sklearn.model_selection import train_test_split, KFold
import sklearn
from sklearn.pipeline import make_pipeline
from numpy.core import _asarray
from sklearn.utils import resample
import time
from utils import ( 
    FrankeFunction, generate_determ_data, create_X,
    KFold_split, predict, compute_betas_ridge, MSE,
    compute_optimal_parameters, find_best_lambda)

def plot_figs_bootstrap_all_lambdas(MSE_train, MSE_test, var, bias, degs, lambdas):
    fig, axs = plt.subplots(2,3)
    fig.suptitle("Plots for analysis of bias-varaince trade-off for ascending values of lambda")
    k = 0
    for i in range(axs.shape[0]): 
        for j in range(axs.shape[1]):
            axs[i,j].set_title(f'Lambda value {lambdas[k]}')
            axs[i,j].plot(degs, MSE_train[k], 'b', label='MSE_train') 
            axs[i,j].plot(degs, MSE_test[k], 'r', label='MSE_test')
            axs[i,j].plot(degs, var[k], 'g', label='variance')
            axs[i,j].plot(degs, bias[k], 'y', label='bias')
            axs[i,j].set_xlabel('Polynomial order')
            axs[i,j].set_ylabel('Mean Squared Error')
            axs[i,j].legend()
            k += 1

    plt.show()


def plot_best_lambda_bv(MSE_train, MSE_test, var, bias, degs, lambda_):
    
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
    axs[1].set_ylabel('bias/variance')
    axs[1].legend()

    plt.show()

def plot_figs_kFold(MSE_train, MSE_test, degs):
    fig, axs = plt.subplots(1,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0].plot(degs, MSE_train[5], 'b', label='MSE_train') 
    axs[0].plot(degs, MSE_test[5], 'r', label='MSE_test')
    axs[0].set_xlabel('Polynomial order')
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].legend()

    axs[1].plot(degs, MSE_train[0], 'b', label='MSE_train') 
    axs[1].plot(degs, MSE_test[0], 'r', label='MSE_test')
    axs[1].set_xlabel('Polynomial order')
    axs[1].set_ylabel('Mean Squared Error')
    axs[1].legend()

    plt.show() 

def Ridge_reg_boot(x, y, lambdas, z=None, n_points=20, degrees=10, n_boots=100, n_lambdas=6, scaling=False, noisy=True, r_seed=79): 
    np.random.seed(r_seed)
    if z == None:
        z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    

    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias_ = np.zeros((n_lambdas, degrees))
    variance_ = np.zeros((n_lambdas, degrees))

    for k in range(len(lambdas)): 
        print(f'Lamda value:{lambdas[k]}')
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

            for boot in range(n_boots):

                X_, z_ = resample(x_train, z_train_centered)
                betas_ridge  = compute_betas_ridge(X_, z_, lambdas[k])

                z_pred_train = predict(x_train, betas_ridge, z_train_mean)
                z_pred_test = predict(x_test, betas_ridge, z_train_mean)

                pred_train_avg[boot, :] = z_pred_train
                pred_test_avg[boot, :] = z_pred_test 

            MSE_train_list[degree-1] = np.mean(np.mean((z_train-pred_train_avg)**2, axis=0, keepdims=True))#training_error/n_boots
            MSE_test_list[degree-1] = np.mean(np.mean((z_test-pred_test_avg)**2, axis=0, keepdims=True))#test_error/n_boots
            bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
            variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))  
            polydegree[degree-1] = degree   
        
        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list
        bias_[k] = bias
        variance_[k] = variance

    return MSE_train, MSE_test, bias_, variance_, polydegree

def Ridge_reg_kFold(x, y, lambdas, z =None, n_points=20, degrees=10, folds=5, n_lambdas=6, scaling=False, noisy=True, r_seed=79):
    np.random.seed(r_seed)
    if z == None: 
        z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    z=z.ravel()
    train_ind, test_ind = KFold_split(z=z, k=folds)

    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    polydegree = np.zeros(degrees)

    for k in range(len(lambdas)): 
        print(f'Lamda value:{lambdas[k]}')
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
       
        for degree in range(1, degrees+1): 
            print(f'Polynomial degree {degree}')

            training_error = 0 
            test_error = 0 

            for train_indx, test_indx in zip(train_ind, test_ind):
                
                x_train, z_train = X[train_indx, :int((degree+1)*(degree+2)/2)], z[train_indx]
                x_test, z_test = X[test_indx, :int((degree+1)*(degree+2)/2)], z[test_indx]
                if scaling:
                    x_train_mean = np.mean(x_train, axis=0) 
                    z_train_mean = np.mean(z_train, axis=0)  
                    x_train -= x_train_mean
                    x_test -= x_train_mean
                    z_train_centered = z_train - z_train_mean
                else: 
                    z_train_centered = z_train
                    z_train_mean = 0 

                betas = compute_betas_ridge(x_train, z_train_centered, lambdas[k])
                
                z_pred_train = predict(x_train, betas, z_train_mean)
                z_pred_test = predict(x_test, betas, z_train_mean)
                training_error += MSE(z_train, z_pred_train)
                test_error += MSE(z_test, z_pred_test)

            MSE_train_list[degree-1] = training_error/folds
            MSE_test_list[degree-1] = test_error/folds 
            polydegree[degree-1] = degree

        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list

    return MSE_train, MSE_test, polydegree

if __name__ == '__main__':
    n_points = 20
    noisy = True 
    n_lambdas = 6
    x,y = generate_determ_data(n_points)
    lambdas = np.logspace(-12,-3,n_lambdas)
    #MSE_train, MSE_test, deg = Ridge_reg_kFold(x,y,lambdas=lambdas, folds=10, r_seed=79, scaling=False)
    MSE_train, MSE_test, bias_, variance_, deg = Ridge_reg_boot(x, y, lambdas=lambdas, r_seed=79, n_points=20, n_boots=100, degrees=12) 
    
    lam, index = find_best_lambda(lambdas, MSE_test)
    
    plot_figs_bootstrap_all_lambdas(MSE_train, MSE_test, variance_, bias_, deg, lambdas)
    plot_best_lambda_bv(MSE_test[index], variance_[index], bias_[index], deg, lam)
    
    
    #plot_figs_kFold(MSE_train, MSE_test, deg)

    # good random_seeds = [79, 227