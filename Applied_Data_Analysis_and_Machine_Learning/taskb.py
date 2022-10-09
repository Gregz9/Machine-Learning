import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
import sklearn
from numpy.core import _asarray
from sklearn.metrics import mean_squared_error
from utils import ( 
    FrankeFunction, generate_random_data, create_X, generate_determ_data, 
    compute_optimal_parameters, predict, R2, MSE)

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
    


def perform_OLS_regression(n_points=20, degrees=10, r_seed=79, noisy=True, scaling=True): 
    np.random.seed(r_seed)
    x, y = generate_determ_data(n_points)
    z = FrankeFunction(x,y, noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    R2_train_list = np.empty(degrees)
    R2_test_list = np.empty(degrees)
    betas_list = []
    preds_cn = []

    for degree in range(1, degrees+1): 
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

        beta_SVD_cn = compute_optimal_parameters(x_train, z_train_centered)
        betas_list.append(beta_SVD_cn)
        # Shifted intercept for use when data is not centered
        #intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)
        
        preds_visualization_cn = predict(X[:, :int((degree+1)*(degree+2)/2)], beta_SVD_cn, z_train_mean)
        preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
        preds_cn.append(preds_visualization_cn)

        preds_train_cn = predict(x_train, beta_SVD_cn, z_train_mean)
        preds_test_cn = predict(x_test, beta_SVD_cn, z_train_mean)

        MSE_train_list[degree-1] = MSE(z_train, preds_train_cn)
        MSE_test_list[degree-1] = MSE(z_test, preds_test_cn)
        R2_train_list[degree-1] = R2(z_train, preds_train_cn)
        R2_test_list[degree-1] = R2(z_test, preds_test_cn)

    return betas_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list,  preds_cn, x, y, z

(betas, MSE_train, MSE_test, 
R2_train, R2_test, preds_cn, x, y, z) = perform_OLS_regression(scaling=True, noisy=True, degrees=11, r_seed=79)

#print(MSE_train)
plot_figs(betas, MSE_train, R2_train, MSE_test, R2_test)
visualize_prediction(x,y,z,preds_cn[4], 5)
