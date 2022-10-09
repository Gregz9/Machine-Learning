

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from imageio.v2 import imread
from numpy.core import _asarray
from sklearn.metrics import mean_squared_error
from utils import ( 
    FrankeFunction, compute_optimal_parameters2, create_X, generate_determ_data, 
    compute_optimal_parameters, predict, R2, MSE, load_and_scale_terrain)

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

def visualize_prediction(x,y,z,pred_vis, order):
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
    


def perform_OLS_regression(n_points=300, n=10, r_seed=79, noisy=True, scaling=True): 
    np.random.seed(r_seed)
    
    terrain_file = ('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_2.tif')
    terrain,N = load_and_scale_terrain(terrain_file)
    x, y = generate_determ_data(N)
    z = terrain[:N,:N]

    MSE_train_list = []
    MSE_test_list = []
    R2_train_list = []
    R2_test_list = []
    betas_list = []
    preds_cn = []

    for i in range(1, n+1): 
        X = create_X(x,y,i)

        if scaling:
            X = np.delete(X, 0, axis=1)
            X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2)
            
            #Centering datasets
            x_train_mean = np.mean(X_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)     

            # Using centered values of X and y to compute parameters beta
            X_train_centered = X_train - x_train_mean
            z_train_centered = z_train - z_train_mean
            X_test_centered  = X_test - x_train_mean 

            beta_SVD_cn = compute_optimal_parameters2(X_train_centered, z_train_centered)
            betas_list.append(beta_SVD_cn)
            # Shifted intercept for use when data is not centered
            intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)
            
            preds_visualization_cn = predict(X, beta_SVD_cn, intercept)
            preds_visualization_cn = preds_visualization_cn.reshape(N, N)
            preds_cn.append(preds_visualization_cn)

            preds_train_cn = predict(X_train_centered, beta_SVD_cn, z_train_mean)
            preds_test_cn = predict(X_test_centered, beta_SVD_cn, z_train_mean)

            MSE_train_list.append(MSE(z_train, preds_train_cn))
            MSE_test_list.append(MSE(z_test, preds_test_cn))
            R2_train_list.append(R2(z_train, preds_train_cn))
            R2_test_list.append(R2(z_test, preds_test_cn))

        elif not scaling: 
            X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2)
            x_train_mean, z_train_mean, intercept = 0, 0, 0
                
            beta_SVD_cn = compute_optimal_parameters(X_train, z_train)
            betas_list.append(beta_SVD_cn)

            preds_visualization_cn = predict(X, beta_SVD_cn)
            preds_visualization_cn = preds_visualization_cn.reshape(N, N)
            preds_cn.append(preds_visualization_cn)

            preds_train_cn = predict(X_train, beta_SVD_cn)
            preds_test_cn = predict(X_test, beta_SVD_cn)

            MSE_train_list.append(MSE(z_train, preds_train_cn))
            MSE_test_list.append(MSE(z_test, preds_test_cn))
            R2_train_list.append(R2(z_train, preds_train_cn))
            R2_test_list.append(R2(z_test, preds_test_cn))

    return betas_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list,  preds_cn, x, y, z

(betas, MSE_train, MSE_test, 
R2_train, R2_test, preds_cn, x, y, z) = perform_OLS_regression(scaling=True, r_seed=79)

#print(MSE_train)
plot_figs(betas, MSE_train, R2_train, MSE_test, R2_test)
#visualize_prediction(x,y,z,preds_cn[4], 4)