from ensurepip import bootstrap
from operator import ge
import numpy as np 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from utils import generate_determ_data, find_best_lambda
from plot_functions import (plot_figs_bootstrap_all_lambdas, plot_kfold_figs_for_k, plot_kFold_figs_for_L, 
                            plot_compare_bootstrap_OLS_Ridge, plot_figs_kFold_compare_OLS_Ridge, compare_all_predictions,
                            show_prediction, compare_bootstrap_MSE, plot_figs_single_run)
from Regression import Lasso_reg_boot, Lasso_reg_kFold, OLS_reg_boot, OLS_reg_kFold, Ridge_reg_boot, Ridge_reg_kFold, OLS_reg, Ridge_reg, Lasso_reg


def task_f(n_points=20, n_lambdas=6, r_seed=79, n_boots=100, degrees=12,       
            noisy=True, centering=True, compare=False, kfold_for_all_lam=False):

    x,y = generate_determ_data(n_points)
    lambdas = np.logspace(-8,-2,n_lambdas)


    MSE_train_boot, MSE_test_boot, bias_, variance_, deg = Lasso_reg_boot(x, y, lambdas_=lambdas, r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, centering=centering) 
    lam_L, index_L = find_best_lambda(lambdas, MSE_test_boot)

    plot_figs_bootstrap_all_lambdas(MSE_train_boot, MSE_test_boot, variance_, bias_, deg, lambdas)

    MSE_train_boot_best, MSE_test_boot_best, bias__best, variance__best, deg = Lasso_reg_boot(x, y, lambdas_=np.array([lam_L]), r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, centering=centering, find_best=True) 

    folds = [5,8,10] 
    MSE_train_folds_L = np.empty((len(folds), degrees))
    MSE_test_folds_L = np.empty((len(folds), degrees))

    for i in range(len(folds)):
        
        MSE_train, MSE_test, deg = Lasso_reg_kFold(x,y,lambdas_=np.array([lam_L]), degrees=degrees, folds=folds[i], r_seed=r_seed, centering=centering)
        MSE_train_folds_L[i], MSE_test_folds_L[i] = MSE_train, MSE_test

    plot_kFold_figs_for_L(MSE_train_folds_L, MSE_test_folds_L, deg, folds)
    if kfold_for_all_lam: 
        MSE_train_all_folds, MSE_test_all_folds, deg = Lasso_reg_kFold(x,y,lambdas_=lambdas, degrees=degrees, folds=folds[len(folds)-1], r_seed=r_seed, centering=centering)
        plot_kfold_figs_for_k(MSE_train_all_folds, MSE_test_all_folds, deg, lambdas)
    
    if compare:

        _, MSE_test_ols, bias_ols, var_ols, _ = OLS_reg_boot(x,y,n_points=n_points, degrees=degrees, 
                                                            n_boots=n_boots, noisy=noisy, r_seed=r_seed, scaling=centering) 

        _, MSE_test_ridge, bias_ridge, var_ridge, _ = Ridge_reg_boot(x,y, lambdas=lambdas, n_points=n_points, degrees=degrees, n_boots=n_boots,
                                                                    noisy=noisy, r_seed=r_seed, scaling=centering)

        lam_R, index_R = find_best_lambda(lambdas, MSE_test_ridge)

        plot_compare_bootstrap_OLS_Ridge(MSE_test_ridge[index_R], var_ridge[index_R], bias_ridge[index_R], lam_R, MSE_test_ols, var_ols, 
                                        bias_ols, MSE_test_boot_best[0], variance__best[0], bias__best[0], lam_L, deg)

        compare_bootstrap_MSE(MSE_test_ols, MSE_test_ridge[index_R], MSE_test_boot_best[0], deg)

        MSE_train_folds_R = np.empty((len(folds), degrees))
        MSE_test_folds_R = np.empty((len(folds), degrees))

        MSE_train_folds_O = np.empty((len(folds), degrees))
        MSE_test_folds_O = np.empty((len(folds), degrees))

        for i in range(len(folds)):
            MSE_train_O, MSE_test_O, pol = OLS_reg_kFold(x,y,n_points=n_points, noisy=noisy, degrees=degrees, 
                                                        r_seed=r_seed, folds=folds[i], scaling=centering)

            MSE_train_R, MSE_test_R, _ = Ridge_reg_kFold(x,y, lambdas=np.array([lam_R]), n_points=n_points, noisy=noisy, degrees=degrees, 
                                                        r_seed=r_seed, folds=folds[i], scaling=centering)

            MSE_train_folds_O[i], MSE_test_folds_O[i] = MSE_train_O, MSE_test_O
            MSE_train_folds_R[i], MSE_test_folds_R[i] = MSE_train_R, MSE_test_R


        plot_figs_kFold_compare_OLS_Ridge(MSE_train_folds_R, MSE_test_folds_R, MSE_train_folds_O, 
                                            MSE_test_folds_O, MSE_train_folds_L, MSE_test_folds_L, deg, folds)

      
        (betas, MSE_train, MSE_test, 
        R2_train, R2_test, preds_cn, x, y, z) = OLS_reg(x, y, n_points=n_points, scaling=centering, noisy=noisy, degrees=degrees, r_seed=r_seed)


        (L_betas, L_MSE_train, L_MSE_test, 
        L_R2_train, L_R2_test, L_preds_cn, x, y, z) = Lasso_reg(x, y, n_points=n_points, lambda_=lam_L, scaling=centering, noisy=noisy, degrees=degrees, r_seed=r_seed)

        (R_betas, R_MSE_train, R_MSE_test, 
        R_R2_train, R_R2_test, R_preds_cn, x, y, z) = Ridge_reg(x, y, n_points=n_points, lambda_=lam_R, scaling=centering, noisy=noisy, degrees=degrees, r_seed=r_seed)

        plot_figs_single_run(MSE_train, MSE_test, R2_train, R2_test, betas, 'OLS')
        plot_figs_single_run(R_MSE_train, R_MSE_test, R_R2_train, R_R2_test, R_betas, 'Ridge')
        plot_figs_single_run(L_MSE_train, L_MSE_test, L_R2_train, L_R2_test, L_betas, 'Lasso')
        
        compare_all_predictions(x,y,z,preds_cn[4], R_preds_cn[4], L_preds_cn[4], 5)
        compare_all_predictions(x,y,z,preds_cn[6], R_preds_cn[6], L_preds_cn[6], 7)
        compare_all_predictions(x,y,z,preds_cn[8], R_preds_cn[8], L_preds_cn[8], 9)
        compare_all_predictions(x,y,z,preds_cn[10], R_preds_cn[10], L_preds_cn[10], 11)

        show_prediction(x,y,preds_cn[4], 5, 'OLS')
        show_prediction(x,y,preds_cn[6], 7, 'OLS')
        show_prediction(x,y,preds_cn[8], 9, 'OLS')
        show_prediction(x,y,preds_cn[10], 11, 'OLS')

        show_prediction(x,y,R_preds_cn[4], 5, 'Ridge')
        show_prediction(x,y,R_preds_cn[6], 7, 'Ridge')
        show_prediction(x,y,R_preds_cn[8], 9, 'Ridge')
        show_prediction(x,y,R_preds_cn[10], 11, 'Ridge')

        show_prediction(x,y,R_preds_cn[4], 5, 'Lasso')
        show_prediction(x,y,R_preds_cn[6], 7, 'Lasso')
        show_prediction(x,y,R_preds_cn[8], 9, 'Lasso')
        show_prediction(x,y,R_preds_cn[10], 11, 'Lasso')        




    # good random_seeds = [79, 227"""

task_f(compare=True)