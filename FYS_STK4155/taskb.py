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
from sklearn.metrics import mean_squared_error
from utils import ( 
    FrankeFunction, generate_random_data, create_X, generate_design_matrix, 
    compute_optimal_parameters, predict, R2, MSE)

def perform_OLS_regression(n_points=40, n=5, seed=None): 

    x, y = generate_random_data(n_points, seed)
    z = FrankeFunction(x,y)

    MSE_train_list = []
    MSE_test_list = []
    R2_train_list = []
    R2_test_list = []
    betas_list = []
    preds_cn = []

    for i in range(1, n+1): 
            
        #X = generate_design_matrix(x, y, i, intercept=False)
        X = create_X(x,y,i)
        X = np.delete(X, 0, axis=1)
        X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2, random_state=seed)
        
        #Centering datasets
        x_train_mean = np.mean(X_train, axis=0) 
        z_train_mean = np.mean(z_train, axis=0)     

        # Using centered values of X and y to compute parameters beta
        X_train_centered = X_train - x_train_mean
        z_train_centered = z_train - z_train_mean
        X_test_centered = X_test - x_train_mean 
        

        beta_SVD_cn = compute_optimal_parameters(X_train_centered, z_train_centered)
        betas_list.append(beta_SVD_cn)

        intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)

        preds_visualization_cn = predict(X, beta_SVD_cn, intercept)

        preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
        preds_cn.append(preds_visualization_cn)

        preds_train_cn = predict(X_train_centered, beta_SVD_cn, z_train_mean)
        #print('Without intercept', np.sum(predict(X_train_centered, beta_SVD_cn)))
        #print('With intercept', np.sum(predict(X_train_centered, beta_SVD_cn, z_train_mean)))

        preds_test_cn = predict(X_test_centered, beta_SVD_cn, z_train_mean)

        MSE_train_list.append(MSE(z_train, preds_train_cn))
        MSE_test_list.append(MSE(z_test, preds_test_cn))
        
        R2_train_list.append(R2(z_train, preds_train_cn))
        R2_test_list.append(R2(z_test, preds_test_cn))

    return betas_list, preds_cn, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list

def plot_figs(*args):
    
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
    x = [i for i in range(1, len(args[0])+1)]
    y = np.zeros((len(x)))

    beta_matrix = np.zeros((len(args[0]), 5))
    for i in range(beta_matrix.shape[0]): 
        for j in range(len(args[0][i])):
            if j == 5:
                break
            beta_matrix[i][j] = args[0][i][j]
    for k in range(5): 
        axs[0,0].plot(x, [beta_matrix[i,k] for i in range(len(args[0]))], color_list[k], label=f'beta{k+1}')
    axs[0,0].plot(x, y, 'k--', label='x-axis')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Beta values')
    axs[0,0].legend()

    axs[0,1].plot(x, args[1], 'b', label='MSE_train') 
    axs[0,1].plot(x, args[3], 'r', label='MSE_test')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('Mean Squared Error')
    axs[0,1].legend()

    axs[1,0].plot(x, args[2], 'g', label='R2_train')
    axs[1,0].plot(x, args[4], 'y', label='R2_test')
    axs[1,0].set_xlabel('Polynomial order')
    axs[1,0].set_ylabel('R2 Score')
    axs[1,0].legend()
    plt.show() 
    # ---------------------------------------------------------------------------------- #
    x, y = generate_random_data(args[6], seed=args[7])
    z = FrankeFunction(x,y)
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    surf = ax.plot_surface(x, y, args[5][4], cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Frankes's function")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # -----------------------------------------------------------------------------------""
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Polynomial fit of n-th order")
    # Add a color bar which maps values to colors 
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
data_size = 20
betas, preds_cn, MSE_train, MSE_test, R2_train, R2_test = perform_OLS_regression(data_size ,n=10, seed=9)

#print(MSE_train)
plot_figs(betas, MSE_train, R2_train, MSE_test, R2_test, preds_cn, data_size, 9)

