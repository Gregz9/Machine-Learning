import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils import ( generate_determ_data, load_and_scale_terrain, find_best_lambda)
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
    
    #Starting off by loading a part of the terrain from the tif file
    terrain,N = load_and_scale_terrain(terrain_file)
    x, y = generate_determ_data(n_points)
    ter = terrain[:20,:20]
    lambdas = np.logspace(-8,-2,n_lambdas)

    # First performing bootstrap
    (OLS_boot_MSE_train, OLS_boot_MSE_test, 
    OLS_boot_bias, OLS_boot_var, degs) = OLS_reg_boot(x, y, n_points=n_points, degrees=degrees, n_boots=n_boots, 
                                                        r_seed=r_seed, scaling=centering)

    (Ridge_boot_MSE_train, Ridge_boot_MSE_test, 
    Ridgr_boot_bias_, Ridge_boot_var, _) = Ridge_reg_boot(x, y, lambdas=lambdas, r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, scaling=centering) 

    lambda_Ridge, index_ridge = find_best_lambda(lambdas, Ridge_boot_MSE_train)
    
    (Lasso_boot_MSE_train, Lasso_boot_MSE_test, 
    Lasso_boot_bias_, Lasso_boot_var, _) = Lasso_reg_boot(x, y, lambdas_=lambdas, r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, centering=centering)

    lambda_Lasso, index_lasso = find_best_lambda(lambdas, Lasso_boot_MSE_test)

    (best_lasso_MSE_boot_train, best_lasso_MSE_boot_test, 
    best_lasso_boot_bias, best_lasso_boot_var, _) = Lasso_reg_boot(x, y, lambdas_=np.array([lambda_Lasso]), r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, centering=centering, find_best=True) 
    
    # Using "optimal"-lambda values we perform an analysis of the MSE using kFold cross validation
    # For the purpose of saving both valuable time and computational resources,an analysis of MSE 
    # through cross validation for varying values of lambda won't be performed when working with real 
    # data
    folds = [5,8,10]
    MSE_train_folds_O = np.empty((len(folds), degrees))
    MSE_test_folds_O = np.empty((len(folds), degrees))

    MSE_train_folds_R = np.empty((len(folds), degrees))
    MSE_test_folds_R = np.empty((len(folds), degrees))

    MSE_train_folds_L = np.empty((len(folds), degrees))
    MSE_test_folds_L = np.empty((len(folds), degrees))
    
    for i in range(len(folds)):
        MSE_train_O, MSE_test_O, _ = OLS_reg_kFold(x,y,n_points=n_points, noisy=noisy, degrees=degrees, r_seed=r_seed, folds=folds[i], scaling=centering)
        MSE_train_R, MSE_test_R, _ = Ridge_reg_kFold(x,y,lambdas=np.array([lambda_Ridge]), degrees=degrees, folds=folds[i], r_seed=r_seed, scaling=centering)
        MSE_train_L, MSE_test_L, _ = Lasso_reg_kFold(x,y,lambdas_=np.array([lambda_Lasso]), degrees=degrees, folds=folds[i], r_seed=r_seed, centering=centering)

        MSE_train_folds_O[i], MSE_test_folds_O[i] = MSE_train_O, MSE_test_O
        MSE_train_folds_R[i], MSE_test_folds_R[i] = MSE_train_R, MSE_test_R
        MSE_train_folds_L[i], MSE_test_folds_L[i] = MSE_train_L, MSE_test_L

    """(betas, MSE_train, MSE_test, 
    R2_train, R2_test, preds_cn, x, y, z) = OLS_reg(x,y,z=ter, n_points=N, degrees=degrees, r_seed=r_seed, noisy=noisy, scaling=True)
    plot_figs_single_run(MSE_train, MSE_test, R2_train, R2_test, betas, 'OLS')
    visualize_terrain(x,y,ter, preds_cn[degrees-1],degrees)"""

terrain_file = ('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_2.tif')
task_g(terrain_file)