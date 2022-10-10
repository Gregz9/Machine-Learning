import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random

def compare_prediction(x,y,z,pred_vis, order):
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Frankes's function")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # -----------------------------------------------------------------------------------""
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, pred_vis, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Polynomial fit of {order}-th order")
    # Add a color bar which maps values to colors 
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_figs_single_run(MSE_train, MSE_test, R2_train, R2_test, beta_values):
    
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
    x = [i for i in range(1, len(beta_values)+1)]
    y = np.zeros((len(x)))

    beta_matrix = np.zeros((len(beta_values), 5))
    for i in range(beta_matrix.shape[0]): 
        for j in range(len(beta_values[i])):
            if j == 5:
                break
            beta_matrix[i][j] = beta_values[i][j]
    for k in range(5): 
        axs[0,0].plot(x, [beta_matrix[i,k] for i in range(len(beta_values))], color_list[k], label=f'beta{k+1}')
    axs[0,0].plot(x, y, 'k--', label='x-axis')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Beta values')
    axs[0,0].legend()

    axs[0,1].plot(x, MSE_train, 'b', label='MSE_train') 
    axs[0,1].plot(x, MSE_test, 'r', label='MSE_test')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('Mean Squared Error')
    axs[0,1].legend()

    axs[1,0].plot(x, R2_train, 'g', label='R2_train')
    axs[1,0].plot(x, R2_test, 'y', label='R2_test')
    axs[1,0].set_xlabel('Polynomial order')
    axs[1,0].set_ylabel('R2 Score')
    axs[1,0].legend()
    plt.show() 
    # ---------------------------------------------------------------------------------- #

def plot_OLS_figs_task_C(MSE_train, MSE_test, var, bias, degs):
    
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

def plot_kfold_figs_for_k(MSE_train, MSE_test, polydegrees, lambdas_ ,fold=10):
    fig, axs = plt.subplots(2,3)
    fig.suptitle(f'MSE of lasso regression kfold witj = {fold} and varying values of lambda parameter')
    k = 0
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):

            axs[i,j].plot(polydegrees, MSE_train[k], 'g', label='MSE_train')
            axs[i,j].plot(polydegrees, MSE_test[k], 'b', label='MSE_test')
            axs[i,j].set_xlabel('Polynomial_order')
            axs[i,j].set_ylabel('MSE')
            axs[i,j].set_title(f'Lambda: {lambdas_[k]}')
            axs[i,j].legend()
            k += 1

    plt.show()


def plot_kFold_figs_for_L(MSE_train, MSE_test, degs, folds, MSE_test_SKL=[]):
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('MSE values for varying values of fold-splits')
    
    k = 0
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            if k == len(folds): 
                break
            axs[i,j].plot(degs, MSE_train[k], 'b', label='MSE_train') 
            axs[i,j].plot(degs, MSE_test[k], 'r', label='MSE_test')
            if len(MSE_test_SKL) > 0: 
                axs[i,j].plot(degs, MSE_test_SKL[k], 'k--', label='MSE_test_scikit')
            axs[i,j].set_title(f'{folds[k]}-folds')
            axs[i,j].set_xlabel('Polynomial order')
            axs[i,j].set_ylabel('Mean Squared Error')
            axs[i,j].legend()
            k += 1 

    plt.show() 

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

def plot_compare_bootstrap_OLS(MSE_test_Ridge, var_Ridge, bias_Ridge, degs, lambda_, MSE_test_OLS, var_OLS, bias_OLS):
     
    fig, axs = plt.subplots(1,2)
    
    axs[0].set_title(f"Bias-variance Ridge regression with optimal lamdbda{lambda_}")
    axs[0].plot(degs, MSE_test_Ridge, 'b', label='MSE_test') 
    axs[0].plot(degs, var_Ridge, 'g', label='variance')
    axs[0].plot(degs, bias_Ridge, 'y', label='bias')
    axs[0].set_xlabel('Polynomial order')
    axs[0].set_ylabel('bias/variance')
    axs[0].legend()

    axs[1].set_title(f'Bias-variance for ordniary least squares regression')
    axs[1].plot(degs, MSE_test_OLS, 'b', label='MSE_test') 
    axs[1].plot(degs, var_OLS, 'g', label='variance')
    axs[1].plot(degs, bias_OLS, 'y', label='bias')
    axs[1].set_xlabel('Polynomial order')
    axs[1].set_ylabel('bias/variance')
    axs[1].legend()

    plt.show()


def plot_compare_bootstrap_OLS_Ridge(MSE_test_Ridge, var_Ridge, bias_Ridge, lambda_R, MSE_test_OLS, var_OLS, 
                                    bias_OLS, MSE_test_Lasso, bias_Lasso, var_Lasoo, Lambda_L, degs):
     
    fig, axs = plt.subplots(1,3)
    
    axs[0].set_title(f"Bias-variance Ridge regression with optimal lamdbda{lambda_R}")
    axs[0].plot(degs, MSE_test_Ridge, 'b', label='MSE_test') 
    axs[0].plot(degs, var_Ridge, 'g', label='variance')
    axs[0].plot(degs, bias_Ridge, 'y', label='bias')
    axs[0].set_xlabel('Polynomial order')
    axs[0].set_ylabel('bias/variance')
    axs[0].legend()

    axs[1].set_title(f'Bias-variance for ordniary least squares regression')
    axs[1].plot(degs, MSE_test_OLS, 'b', label='MSE_test') 
    axs[1].plot(degs, var_OLS, 'g', label='variance')
    axs[1].plot(degs, bias_OLS, 'y', label='bias')
    axs[1].set_xlabel('Polynomial order')
    axs[1].set_ylabel('bias/variance')
    axs[1].legend()

    axs[2].set_title(f'Bias-variance for ordniary least squares regression')
    axs[2].plot(degs, MSE_test_Lasso, 'b', label='MSE_test') 
    axs[2].plot(degs, var_Lasoo, 'g', label='variance')
    axs[2].plot(degs, bias_Lasso, 'y', label='bias')
    axs[2].set_xlabel('Polynomial order')
    axs[2].set_ylabel('bias/variance')
    axs[2].legend()

    plt.show()


def plot_figs_kFold_compare_OLS(MSE_train_ridge, MSE_test_ridge, MSE_train_OLS, MSE_test_OLS, degs, folds):
    fig, axs = plt.subplots(3,2)
    fig.suptitle('MSE values for training and test data for varying degrees of kfold-splits')
    k = 0
    for i in range(axs.shape[0]): 
        for j in range(axs.shape[1]):
            if j == 0: 
                axs[i,j].set_title(f'{folds[k]}-folds for Rigde regression with optimal beta')
                axs[i,j].plot(degs, MSE_train_ridge[k], 'b', label='MSE_train') 
                axs[i,j].plot(degs, MSE_test_ridge[k], 'r', label='MSE_test')
                axs[i,j].set_xlabel('Polynomial order')
                axs[i,j].set_ylabel('Mean Squared Error')
                axs[i,j].legend()
            elif j == 1:
                axs[i,j].set_title(f'{folds[k]}-folds for OLS regression')
                axs[i,j].plot(degs, MSE_train_OLS[k], 'b', label='MSE_train') 
                axs[i,j].plot(degs, MSE_test_OLS[k], 'r', label='MSE_test')
                axs[i,j].set_xlabel('Polynomial order')
                axs[i,j].set_ylabel('Mean Squared Error')
                axs[i,j].legend()
        k += 1
    plt.show() 