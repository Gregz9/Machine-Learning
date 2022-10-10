import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random



def show_prediction(x,y,pred_vis, order, Reg_type):
    fig= plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, pred_vis, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Polynomial fit of {order}-th order using {Reg_type}")
    
    plt.show()


def compare_2_predictions(x,y,z,pred_vis, order):
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Frankes's function")
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x, y, pred_vis, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Polynomial fit of {order}-th order")
    
    plt.show()

def compare_all_predictions(x, y, z, pred_vis_ols, pred_vis_ridge, pred_vis_lasso, order):
    fig= plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Frankes's function")

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot_surface(x, y, pred_vis_ols, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Polynomial fit of {order}-th order using OLS")

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.plot_surface(x, y, pred_vis_ridge, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Polynomial fit of {order}-th order using Ridge")

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(x, y, pred_vis_lasso, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Polynomial fit of {order}-th order using Lasso")
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


def compare_MSE(MSE_test_ols, MSE_test_ridge, MSE_test_lasso,degs):
    
    fig, axs = plt.subplots(1,1)

    axs.plot(degs, MSE_test_ols, 'b', label='MSE_test OLS') 
    axs.plot(degs, MSE_test_ridge, 'r', label='MSE_test Ridge')
    axs.plot(degs, MSE_test_lasso, 'purple', label='MSE_test Lasso')
    axs.set_xlabel('Polynomial order')
    axs.set_ylabel('Mean Squared Error')
    axs.legend()
    plt.show()

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
                                    bias_OLS, MSE_test_Lasso, var_Lasoo, bias_Lasso, Lambda_L, degs):
     
    fig, axs = plt.subplots(1,3)
    
    axs[0].set_title(f"Bias-variance Ridge with lamdbda{lambda_R}")
    axs[0].plot(degs, MSE_test_Ridge, 'b', label='MSE_test') 
    axs[0].plot(degs, var_Ridge, 'g', label='variance')
    axs[0].plot(degs, bias_Ridge, 'y', label='bias')
    axs[0].set_ylim(0.000, 0.030)
    axs[0].set_xlabel('Polynomial order')
    axs[0].set_ylabel('bias/variance')
    axs[0].legend()

    axs[1].set_title(f'Bias-variance for ordniary least squares regression')
    axs[1].plot(degs, MSE_test_OLS, 'b', label='MSE_test') 
    axs[1].plot(degs, var_OLS, 'g', label='variance')
    axs[1].plot(degs, bias_OLS, 'y', label='bias')
    axs[1].set_ylim(0.000, 0.030)
    axs[1].set_xlabel('Polynomial order')
    axs[1].set_ylabel('bias/variance')
    axs[1].legend()

    axs[2].set_title(f"Bias-variance Lasso with lamdbda{Lambda_L}")
    axs[2].plot(degs, MSE_test_Lasso, 'b', label='MSE_test') 
    axs[2].plot(degs, var_Lasoo, 'g', label='variance')
    axs[2].plot(degs, bias_Lasso, 'y', label='bias')
    axs[2].set_ylim(0.000, 0.030)
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

def plot_figs_kFold_compare_OLS_Ridge(MSE_train_ridge, MSE_test_ridge, MSE_train_OLS, MSE_test_OLS,
                                     MSE_train_Lasso, MSE_test_Lasoo, degs, folds):
    fig, axs = plt.subplots(3,3)
    fig.suptitle('MSE values for training and test data for varying degrees of kfold-splits')
    k = 0
    for i in range(axs.shape[0]): 
        for j in range(axs.shape[1]):
            if j == 0:
                axs[i,j].set_title(f'{folds[k]}-folds for OLS regression')
                axs[i,j].plot(degs, MSE_train_OLS[k], 'b', label='MSE_train') 
                axs[i,j].plot(degs, MSE_test_OLS[k], 'r', label='MSE_test')
                axs[i,j].set_xlabel('Polynomial order')
                axs[i,j].set_ylabel('Mean Squared Error')
                axs[i,j].legend()
            
            elif j == 1: 
                axs[i,j].set_title(f'{folds[k]}-folds for Rigde regression with optimal beta')
                axs[i,j].plot(degs, MSE_train_ridge[k], 'b', label='MSE_train') 
                axs[i,j].plot(degs, MSE_test_ridge[k], 'r', label='MSE_test')
                axs[i,j].set_xlabel('Polynomial order')
                axs[i,j].set_ylabel('Mean Squared Error')
                axs[i,j].legend()

            elif j == 2: 
                axs[i,j].set_title(f'{folds[k]}-folds for Lasso regression with optimal beta')
                axs[i,j].plot(degs, MSE_train_Lasso[k], 'b', label='MSE_train') 
                axs[i,j].plot(degs, MSE_test_Lasoo[k], 'r', label='MSE_test')
                axs[i,j].set_xlabel('Polynomial order')
                axs[i,j].set_ylabel('Mean Squared Error')
                axs[i,j].legend()
 
        k += 1
    plt.show() 