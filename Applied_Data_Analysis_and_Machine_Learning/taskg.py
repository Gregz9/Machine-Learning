import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils import ( 
    FrankeFunction, compute_optimal_parameters2, create_X, generate_determ_data, 
    compute_optimal_parameters, predict, R2, MSE, load_and_scale_terrain)
from Regression import (OLS_reg, OLS_reg_boot, OLS_reg_kFold, Ridge_reg, Ridge_reg_boot, Ridge_reg_kFold,
                         Lasso_reg, Lasso_reg_kFold, Lasso_reg_boot)
from plot_functions import compare_2_predictions, plot_figs_single_run


def visualize_terrain(x,y,z,pred_vis, order):
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.terrain, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.75)
    ax.set_zlabel('Height [km]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Real terrain")

    ax = fig.add_subplot(1, 2, 2,projection='3d')
    ax.plot_surface(x, y, pred_vis, cmap=cm.terrain, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.75)
    ax.set_zlabel('Height [km]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
    ax.set_title(f"Polynomial fit of {order}-th order")
    plt.show()

def task_g(terrain_file, n_points=20, n_lambdas=6, r_seed=79, n_boots=10, degrees=20,       
            noisy=True, centering=True, compare=False, kfold_for_all_lam=False):

    
    

    terrain,N = load_and_scale_terrain(terrain_file)
    x, y = generate_determ_data(N)
    ter = terrain[:N,:N]
    print(len(ter))
    (betas, MSE_train, MSE_test, 
    R2_train, R2_test, preds_cn, x, y, z) = OLS_reg(x,y,z=ter, n_points=N, degrees=degrees, r_seed=r_seed, noisy=noisy, scaling=True)
    plot_figs_single_run(MSE_train, MSE_test, R2_train, R2_test, betas, 'OLS')
    visualize_terrain(x,y,ter, preds_cn[degrees-1],degrees)

terrain_file = ('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_2.tif')
task_g(terrain_file)