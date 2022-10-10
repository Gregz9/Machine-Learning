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