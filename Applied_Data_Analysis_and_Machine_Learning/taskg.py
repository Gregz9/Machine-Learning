import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils import ( generate_determ_data, load_and_scale_terrain, find_best_lambda)
from Regression import (OLS_reg, OLS_reg_boot, OLS_reg_kFold, Ridge_reg, Ridge_reg_boot, Ridge_reg_kFold,
                         Lasso_reg, Lasso_reg_kFold, Lasso_reg_boot)
from plot_functions import (compare_2_predictions, plot_figs_single_run, compare_all_predictions, 
                            show_terrain, plot_figs_bootstrap_all_lambdas, plot_kFold_figs_for_L,
                            plot_compare_bootstrap_OLS_Ridge, compare_bootstrap_MSE, plot_figs_kFold_compare_OLS_Ridge,
                            compare_bootstrap_Ridge_Lasso)


from plot_functions import plot_OLS_figs_task_C as plot_OLS_bootstrap

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

def task_g(terrain_file, n_lambdas=6, r_seed=79, n_boots=10, degrees=20,       
            noisy=True, centering=True, compare=False, kfold_for_all_lam=False):
    
    #Starting off by loading a part of the terrain from the tif file
    terrain,N = load_and_scale_terrain(terrain_file)
    N=100
    n_points=N
    x, y = generate_determ_data(N)
    ter = terrain[:N,:N]
    lambdas = np.logspace(-6,0,n_lambdas)

    # First performing bootstrap
    (OLS_boot_MSE_train, OLS_boot_MSE_test, 
    OLS_boot_bias, OLS_boot_var, degs) = OLS_reg_boot(x, y, n_points=n_points, degrees=degrees, n_boots=n_boots, 
                                                        r_seed=r_seed, scaling=centering)

    (Ridge_boot_MSE_train, Ridge_boot_MSE_test, 
    Ridge_boot_bias_, Ridge_boot_var, _) = Ridge_reg_boot(x, y, z=ter, lambdas=lambdas, r_seed=r_seed, 
                                                                n_boots=n_boots, degrees=degrees, scaling=centering) 

    lambda_Ridge, index_ridge = find_best_lambda(lambdas, Ridge_boot_MSE_train)
    
    (Lasso_boot_MSE_train, Lasso_boot_MSE_test, 
    Lasso_boot_bias_, Lasso_boot_var, _) = Lasso_reg_boot(x, y, z=ter, lambdas_=lambdas, r_seed=r_seed, 
                                                                n_boots=n_boots, degrees=degrees, centering=centering)

    lambda_Lasso, index_lasso = find_best_lambda(lambdas, Lasso_boot_MSE_test)

    (best_lasso_MSE_boot_train, best_lasso_MSE_boot_test, 
    best_lasso_boot_bias, best_lasso_boot_var, _) = Lasso_reg_boot(x, y, z=ter, lambdas_=np.array([lambda_Lasso]), r_seed=r_seed, 
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
        MSE_train_O, MSE_test_O, _ = OLS_reg_kFold(x, y, z=ter, noisy=noisy, degrees=degrees, r_seed=r_seed, folds=folds[i], scaling=centering)
        MSE_train_R, MSE_test_R, _ = Ridge_reg_kFold(x, y, z=ter, lambdas=np.array([lambda_Ridge]), degrees=degrees, folds=folds[i], r_seed=r_seed, scaling=centering)
        MSE_train_L, MSE_test_L, _ = Lasso_reg_kFold(x, y, z=ter, lambdas_=np.array([lambda_Lasso]), degrees=degrees, folds=folds[i], r_seed=r_seed, centering=centering)

        MSE_train_folds_O[i], MSE_test_folds_O[i] = MSE_train_O, MSE_test_O
        MSE_train_folds_R[i], MSE_test_folds_R[i] = MSE_train_R, MSE_test_R
        MSE_train_folds_L[i], MSE_test_folds_L[i] = MSE_train_L, MSE_test_L


    (betas, MSE_train, MSE_test, 
    R2_train, R2_test, preds_cn, x, y, z) = OLS_reg(x, y, z=ter, n_points=n_points, scaling=centering, noisy=noisy, degrees=degrees, r_seed=r_seed)

    (R_betas, R_MSE_train, R_MSE_test, 
    R_R2_train, R_R2_test, R_preds_cn, x, y, z) = Ridge_reg(x, y, z=ter, n_points=n_points, lambda_=lambda_Ridge, scaling=centering, noisy=noisy, degrees=degrees, r_seed=r_seed)
    
    (L_betas, L_MSE_train, L_MSE_test, 
    L_R2_train, L_R2_test, L_preds_cn, x, y, z) = Lasso_reg(x, y, z=ter, n_points=n_points, lambda_=lambda_Lasso, scaling=centering, noisy=noisy, degrees=degrees, r_seed=r_seed)

    plot_OLS_bootstrap(OLS_boot_MSE_train, OLS_boot_MSE_train, OLS_boot_var, OLS_boot_bias, degs)
    plot_figs_bootstrap_all_lambdas(Ridge_boot_MSE_train, Ridge_boot_MSE_test, Ridge_boot_var, Ridge_boot_bias_, degs, lambdas, 'Ridge')
    plot_figs_bootstrap_all_lambdas(Lasso_boot_MSE_train, Lasso_boot_MSE_test, Lasso_boot_var, Lasso_boot_bias_, degs, lambdas, 'Lasso')    
    
    plot_kFold_figs_for_L(MSE_train_folds_O, MSE_test_folds_O, degs, folds, reg_type='OLS')
    plot_kFold_figs_for_L(MSE_train_folds_R, MSE_test_folds_R, degs, folds, reg_type='Ridge') 
    plot_kFold_figs_for_L(MSE_train_folds_L, MSE_test_folds_L, degs, folds, reg_type='Lasso') 

    plot_compare_bootstrap_OLS_Ridge(Ridge_boot_MSE_test[index_ridge], Ridge_boot_var[index_ridge], Ridge_boot_bias_[index_ridge],
                                    lambda_Ridge, OLS_boot_MSE_test, OLS_boot_var, OLS_boot_bias, Lasso_boot_MSE_test[index_lasso],
                                    Lasso_boot_var[index_lasso], Lasso_boot_bias_[index_lasso], lambda_Lasso, degs)
    
    compare_bootstrap_MSE(OLS_boot_MSE_test, Ridge_boot_MSE_test[index_ridge], Lasso_boot_MSE_test[index_lasso], degs)

    compare_bootstrap_Ridge_Lasso(Ridge_boot_MSE_test[index_ridge], Lasso_boot_MSE_test[index_lasso], degs)

    plot_figs_kFold_compare_OLS_Ridge(MSE_train_folds_R, MSE_test_folds_R, MSE_train_folds_O, MSE_test_folds_O, 
                                        MSE_train_folds_L, MSE_test_folds_L, degs, folds)
   
    plot_figs_single_run(MSE_train, MSE_test, R2_train, R2_test, betas, 'OLS')
    plot_figs_single_run(R_MSE_train, R_MSE_test, R_R2_train, R_R2_test, R_betas, 'Ridge')
    plot_figs_single_run(L_MSE_train, L_MSE_test, L_R2_train, L_R2_test, L_betas, 'Lasso')
    
    compare_all_predictions(x, y, ter, preds_cn[10], R_preds_cn[10], L_preds_cn[10], 11, cm.terrain)
    compare_all_predictions(x, y, ter, preds_cn[13], R_preds_cn[13], L_preds_cn[13], 14, cm.terrain)
    compare_all_predictions(x, y, ter, preds_cn[16], R_preds_cn[16], L_preds_cn[16], 17, cm.terrain)
    compare_all_predictions(x, y, ter, preds_cn[19], R_preds_cn[19], L_preds_cn[19], 20, cm.terrain)

    show_terrain(x, y, preds_cn[10], 11, 'OLS')
    show_terrain(x, y, preds_cn[13], 14, 'OLS')
    show_terrain(x, y, preds_cn[16], 17, 'OLS')
    show_terrain(x, y, preds_cn[19], 20, 'OLS')

    show_terrain(x, y, R_preds_cn[10], 11, 'Ridge')
    show_terrain(x, y, R_preds_cn[13], 14, 'Ridge')
    show_terrain(x, y, R_preds_cn[16], 17, 'Ridge')
    show_terrain(x, y, R_preds_cn[19], 20, 'Ridge')

    show_terrain(x, y, R_preds_cn[10], 11, 'Lasso')
    show_terrain(x, y, R_preds_cn[13], 14, 'Lasso')
    show_terrain(x, y, R_preds_cn[16], 17, 'Lasso')
    show_terrain(x, y, R_preds_cn[19], 20, 'Lasso')        

terrain_file = ('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_2.tif')
task_g(terrain_file)