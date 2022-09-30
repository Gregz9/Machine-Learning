import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
import sklearn
from numpy.core import _asarray
from sklearn.utils import resample
from utils import ( 
    FrankeFunction, generate_random_data, create_X, generate_design_matrix, 
    compute_optimal_parameters, predict, MSE)

def OLS_boot_reg(n_points=20, degrees=5, n_boots=10, seed=None): 

    x, y = generate_random_data(n_points, seed)
    z = FrankeFunction(x,y,noise=True)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    betas_list = []
    preds_cn = np.empty((n_points**2,degrees))
    bias = np.zeros(degrees)
    variance = np.zeros(degrees)


    for degree in range(1, degrees+1): 
            
        X = generate_design_matrix(x,y,degree)
        MSE_train_avg = np.empty((n_boots))
        MSE_test_avg = np.empty((n_boots))
        #betas_avg = np.empty((3*degree, n_boots))
        #preds_avg = np.empty((n_points**2,n_boots))


        X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2)#, random_state=seed)
        z_pred_train = np.empty((z_train.shape[0], n_boots))
        z_pred_test = np.empty((z_test.shape[0], n_boots))
        for j in range(n_boots): 
            # Bootstrap resampling of datasets after split
            X_train, z_train = resample(X_train, z_train)
            #X_test, z_test = resample(X_test, z_test)

            #Centering datasets
            x_train_mean = np.mean(X_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)     

            # Using centered values of X and y to compute parameters beta
            X_train_centered = X_train - x_train_mean
            z_train_centered = z_train - z_train_mean
            X_test_centered = X_test - x_train_mean 

            beta_SVD_cn = compute_optimal_parameters(X_train, z_train)
            #betas_avg[:, j] = (beta_SVD_cn)
           

            intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)

            #preds_visualization_cn = predict(X, beta_SVD_cn, z_train_mean)
            #preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
            #preds_avg[:, j] = preds_visualization_cn

            z_pred_train[:, j] = predict(X_train, beta_SVD_cn)#, z_train_mean) 
            z_pred_test[:, j] = predict(X_test, beta_SVD_cn)#, z_train_mean)

            MSE_train_avg[j] = MSE(z_train, z_pred_train[:, j])
            #MSE_test_avg[j] = MSE(z_test, z_pred_test[:, j])
        
        z_test = z_test.reshape((len(z_test), 1))
        #betas_list.append(np.mean(betas_avg, axis=1))
        MSE_train_list[degree-1] = np.mean(MSE_train_avg)#, keepdims=True)
        MSE_test_list[degree-1] = np.mean(np.mean((z_pred_test-z_test)**2,axis=1, keepdims=True))
        bias[degree-1] = np.mean((z_test - np.mean(z_pred_test, axis=1, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(z_pred_test, axis=1, keepdims=True))
        #preds_cn[:, degree-1] = np.mean(preds_avg, axis=1)
        
    
    return bias, variance, MSE_train_list, MSE_test_list

def plot_OLS_boot_figs(*args):
    
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
    x = [i for i in range(1, len(args[0])+1)]
    y = np.zeros((len(x)))

    axs[0,0].plot(x, args[0], 'b', label='MSE_train') 
    axs[0,0].plot(x, args[1], 'r', label='MSE_test')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Mean Squared Error')
    axs[0,0].legend()
   
    axs[0,1].plot(x, args[1], 'b', label='MSE_test') 
    axs[0,1].plot(x, args[2], 'g', label='variance')
    axs[0,1].plot(x, args[3], 'y', label='bias')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('R2 Score')
    axs[0,1].legend()
    
    plt.show() 

import time
start_time =time.time()

#_,_, MSE_train, MSE_test = OLS_boot_reg(n_points=40, degrees=12, n_boots=20, seed=9)
#bias, var, MSE_train, MSE_test = OLS_boot_reg(n_points=40, degrees=11, n_boots=12, seed=47)
#bias, var, MSE_train, MSE_test = OLS_boot_reg(n_points=40, degrees=11, n_boots=25, seed=245)
#bias, var, MSE_train, MSE_test = OLS_boot_reg(n_points=40, degrees=11, n_boots=25, seed=1911)
bias, var, MSE_train, MSE_test = OLS_boot_reg(n_points=40, degrees=11, n_boots=20, seed=2546)


plot_OLS_boot_figs(MSE_train, MSE_test, var, bias)
print("--- %s seconds ---" %(time.time()- start_time))
