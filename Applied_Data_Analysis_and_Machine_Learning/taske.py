import numpy as np 
from utils import generate_determ_data, find_best_lambda
from Regression import Ridge_reg_boot, Ridge_reg_kFold, OLS_reg_kFold, OLS_reg_boot
from plot_functions import (plot_figs_bootstrap_all_lambdas, plot_kFold_figs_for_L, 
                            plot_compare_bootstrap_OLS, plot_figs_kFold_compare_OLS,
                            plot_kfold_figs_for_k)


def task_e(n_points=20, n_lambdas=6, r_seed=79, n_boots=100, degrees=12,       
            noisy=True, centering=True, compare=False, kfold_for_all_lam=False):

    x,y = generate_determ_data(n_points)
    lambdas = np.logspace(-12,-3,n_lambdas)


    MSE_train_boot, MSE_test_boot, bias_, variance_, deg = Ridge_reg_boot(x, y, lambdas=lambdas, r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, scaling=centering) 
    lam, index = find_best_lambda(lambdas, MSE_train_boot)

    plot_figs_bootstrap_all_lambdas(MSE_train_boot, MSE_test_boot, variance_, bias_, deg, lambdas)
    #-----------------------------------------------------------------------------------------------------------------------------#
    folds = [5,8,10] 
    MSE_train_folds_R = np.empty((len(folds), degrees))
    MSE_test_folds_R = np.empty((len(folds), degrees))

    for i in range(len(folds)):
        
        MSE_train, MSE_test, deg = Ridge_reg_kFold(x,y,lambdas=np.array([lam]), degrees=degrees, folds=folds[i], r_seed=r_seed, scaling=centering)
        MSE_train_folds_R[i], MSE_test_folds_R[i] = MSE_train, MSE_test
    print(deg)

    plot_kFold_figs_for_L(MSE_train_folds_R, MSE_test_folds_R, deg, folds)
    if kfold_for_all_lam: 
        MSE_train_all_folds, MSE_test_all_folds, deg = Ridge_reg_kFold(x,y,lambdas=lambdas, degrees=degrees, folds=folds[len(folds)-1], r_seed=r_seed, scaling=centering)
        plot_kfold_figs_for_k(MSE_train_all_folds, MSE_test_all_folds, deg, lambdas)
    #-----------------------------------------------------------------------------------------------------------------------------#
    if compare:
        _, MSE_test_ols, bias_ols, var_ols, _ = OLS_reg_boot(x,y,n_points=n_points, degrees=degrees, 
                                                            n_boots=n_boots, noisy=noisy, r_seed=r_seed, scaling=centering) 
        plot_compare_bootstrap_OLS(MSE_test_boot[index], variance_[index], bias_[index], deg, lam, MSE_test_ols, var_ols, bias_ols)

        MSE_train_folds_O = np.empty((len(folds), degrees))
        MSE_test_folds_O = np.empty((len(folds), degrees))

        for i in range(len(folds)):
            MSE_train, MSE_test, pol = OLS_reg_kFold(x,y,n_points=n_points, noisy=noisy, degrees=degrees, 
                                                        r_seed=r_seed, folds=folds[i], scaling=centering)
            MSE_train_folds_O[i], MSE_test_folds_O[i] = MSE_train, MSE_test

        plot_figs_kFold_compare_OLS(MSE_train_folds_R, MSE_test_folds_R, MSE_train_folds_O, MSE_test_folds_O, deg, folds)

    # good random_seeds = [79, 227
task_e(kfold_for_all_lam=True)

