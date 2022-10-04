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

def OLS_cross_reg(n_points=20, degrees=5, folds=5, scaling=False, noisy=True, r_seed=9, include_wrong_calc=False): 
    np.random.seed(r_seed)
    
    x,y = generate_determ_data(n_points)
    z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees)
    z = z.ravel()
    train_ind, test_ind = KFold_split(z=z, k=folds)

    MSE_train = np.empty(degrees)
    MSE_test = np.empty(degrees)
    bias = np.zeros(degrees)
    variance = np.zeros(degrees)
    polydegree = np.zeros(degrees)

    i, i2 = 3,3
    for degree in range(1, degrees+1): 
        pred_train_avg = []
        pred_test_avg = []
        z_train_set = []
        z_test_set = []
        var_avg = []
        bias_avg = []
        training_error = 0 
        test_error = 0
        print(f'Polynomial degree {degree}')
        
        for train_indx, test_indx in zip(train_ind, test_ind):
            
            x_train, z_train = X[train_indx, :i], z[train_indx]
            x_test, z_test = X[test_indx, :i], z[test_indx]
           
            z_train_set.append(z_train)
            z_test_set.append(z_test)
            
            betas = compute_optimal_parameters(x_train, z_train)
            z_pred_train = predict(x_train, betas)
            z_pred_test = predict(x_test, betas)

            pred_train_avg.append(z_pred_train)
            pred_test_avg.append(z_pred_test)

            training_error += MSE(z_train, z_pred_train)
            test_error += MSE(z_test, z_pred_test)

        i += i2 
        i2 += 1

        if n_points**2%folds != 0:
            testInd = min([test.shape for test in pred_test_avg])
            testInd = [i for i in range(len(pred_test_avg)) if pred_test_avg[i].shape == testInd][0]
            tst = np.concatenate((z_test_set[:testInd] + z_test_set[testInd+1:]))

            trainInd = min([train.shape for train in pred_train_avg])
            trainInd = [i for i in range(len(pred_train_avg)) if pred_train_avg[i].shape == trainInd][0]
            trn = np.concatenate((z_train_set[:trainInd] + z_train_set[trainInd+1:]))

            tst_pred = np.concatenate((pred_test_avg[:testInd] + pred_test_avg[testInd+1:]))
            trn_pred = np.concatenate((pred_train_avg[:trainInd] + pred_train_avg[trainInd+1:]))

            MSE_train[degree-1] =  np.mean(np.mean((trn-trn_pred)**2, axis=0, keepdims=True))              
            MSE_test[degree-1] = np.mean(np.mean((tst-tst_pred)**2, axis=0, keepdims=True)) 
            polydegree[degree-1] = degree
        else: 

            MSE_train[degree-1] = training_error/folds 
            MSE_test[degree-1] = test_error/folds 
            polydegree[degree-1] = degree

    return bias, variance, MSE_train, MSE_test, polydegree


def plot_OLS_boot_figs(*args):
    
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0,0].plot(args[2], args[0], 'b', label='MSE_train') 
    axs[0,0].plot(args[2], args[1], 'r', label='MSE_test')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Mean Squared Error')
    axs[0,0].legend()
    
    plt.show() 

# Good values for the random seed variable r_seed => [2, 3, 17 
# Size of dataset good for the analysis of bias-variance trade-off => 10

bias,var, MSE_train, MSE_test, pol = OLS_cross_reg(n_points=20, degrees=10, r_seed=79, folds=10)

plot_OLS_boot_figs(MSE_train, MSE_test, pol)