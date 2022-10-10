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
    FrankeFunction, generate_determ_data, create_X, compute_optimal_parameters, 
    compute_optimal_parameters_inv, generate_design_matrix, predict, MSE, KFold_split,
)

def plot_OLS_kFold_figs(MSE_train, MSE_test, degs, folds, MSE_test_SKL=[]):
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('MSE values for varying values of fold-splits')
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0,0].plot(degs, MSE_train[0], 'b', label='MSE_train') 
    axs[0,0].plot(degs, MSE_test[0], 'r', label='MSE_test')
    axs[0,0].plot(degs, MSE_test_scikit[0], 'k--', label='MSE_test_scikit')
    axs[0,0].set_title(f'{folds[0]}-folds')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Mean Squared Error')
    axs[0,0].legend()
    
    axs[0,1].plot(degs, MSE_train[1], 'b', label='MSE_train') 
    axs[0,1].plot(degs, MSE_test[1], 'r', label='MSE_test')
    axs[0,1].plot(degs, MSE_test_scikit[1], 'k--', label='MSE_test_scikit')
    axs[0,1].set_title(f'{folds[1]}-folds')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('Mean Squared Error')
    axs[0,1].legend()

    axs[1,0].plot(degs, MSE_train[2], 'b', label='MSE_train') 
    axs[1,0].plot(degs, MSE_test[2], 'r', label='MSE_test')
    axs[1,0].plot(degs, MSE_test_scikit[2], 'k--', label='MSE_test_scikit')
    axs[1,0].set_title(f'{folds[2]}-folds')
    axs[1,0].set_xlabel('Polynomial order')
    axs[1,0].set_ylabel('Mean Squared Error')
    axs[1,0].legend()

    axs[1,1].plot(degs, MSE_train[3], 'b', label='MSE_train') 
    axs[1,1].plot(degs, MSE_test[3], 'r', label='MSE_test')
    axs[1,1].plot(degs, MSE_test_scikit[3], 'k--', label='MSE_test_scikit')
    axs[1,1].set_title(f'{folds[3]}-folds')
    axs[1,1].set_xlabel('Polynomial order')
    axs[1,1].set_ylabel('Mean Squared Error')
    axs[1,1].legend()
    plt.show() 

def OLS_reg_kFold(x,y,z=None,n_points=20, degrees=5, folds=5, scaling=False, noisy=True, r_seed=79): 
    np.random.seed(r_seed)
    
    if z == None: 
        z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    z = z.ravel()

    MSE_train = np.empty(degrees)
    MSE_test = np.empty(degrees)

    polydegree = np.zeros(degrees)

    for degree in range(1, degrees+1): 
        pred_train_avg = []
        pred_test_avg = []
        training_error = 0 
        test_error = 0
        print(f'Polynomial degree {degree}')
        train_ind, test_ind = KFold_split(z=z, k=folds)
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
            
            betas = compute_optimal_parameters(x_train, z_train_centered)
            z_pred_train = predict(x_train, betas, z_train_mean)
            z_pred_test = predict(x_test, betas, z_train_mean)

            pred_train_avg.append(z_pred_train)
            pred_test_avg.append(z_pred_test)
            training_error += MSE(z_train, z_pred_train)
            test_error += MSE(z_test, z_pred_test)

        MSE_train[degree-1] = training_error/folds 
        MSE_test[degree-1] = test_error/folds 
        polydegree[degree-1] = degree

    return MSE_train, MSE_test, polydegree


def OLS_reg_kFold_scikit_learn(x,y,z=None,n_points=20, degrees=5, folds=5, scaling=False, noisy=True, r_seed=79): 
    np.random.seed(r_seed)
    
    if z == None: 
        z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    z = z.ravel()

    MSE_train = np.empty(degrees)
    MSE_test = np.empty(degrees)

    polydegree = np.zeros(degrees)

    for degree in range(1, degrees+1): 
        pred_train_avg = []
        pred_test_avg = []
        training_error = 0 
        test_error = 0
        print(f'Polynomial degree {degree}')
        kFold = KFold(n_splits=folds)
        for train_indx, test_indx in kFold.split(X):
            
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
            
            betas = compute_optimal_parameters(x_train, z_train_centered)
            z_pred_train = predict(x_train, betas, z_train_mean)
            z_pred_test = predict(x_test, betas, z_train_mean)

            pred_train_avg.append(z_pred_train)
            pred_test_avg.append(z_pred_test)
            training_error += MSE(z_train, z_pred_train)
            test_error += MSE(z_test, z_pred_test)

        MSE_train[degree-1] = training_error/folds 
        MSE_test[degree-1] = test_error/folds 
        polydegree[degree-1] = degree

    return MSE_train, MSE_test, polydegree
if __name__ == '__main__': 
    n_points = 20 
    noisy = True
    order = 10 
    x,y = generate_determ_data(n_points)
    folds = [5,6,8,10]
    MSE_train_folds = np.empty((len(folds), order))
    MSE_test_folds = np.empty((len(folds), order))
    MSE_test_scikit = np.empty((len(folds), order))
    for i in range(len(folds)):
        MSE_train, MSE_test, pol = OLS_reg_kFold(x,y,n_points=n_points, noisy=noisy, degrees=order, r_seed=79, folds=folds[i], scaling=True)
        MSE_train_folds[i], MSE_test_folds[i] = MSE_train, MSE_test
        _, MSE_t_sci, _ = OLS_reg_kFold_scikit_learn(x,y,n_points=n_points, noisy=noisy, degrees=order, r_seed=79, folds=folds[i], scaling=True)
        MSE_test_scikit[i] = MSE_t_sci
    
    plot_OLS_kFold_figs(MSE_train_folds, MSE_test_folds, pol, folds, MSE_test_scikit)