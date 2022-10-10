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
from plot_functions import plot_figs_bootstrap_all_lambdas, plot_kfold_figs_for_k, plot_kFold_figs_for_L
from Regression import Lasso_reg_boot, Lasso_reg_kFold, OLS_reg_boot, OLS_reg_kFold, Ridge_reg_boot, Ridge_reg_kFold

#MSE_train, MSE_test, polydegrees, lambdas_ = lasso_reg_kFold(n_points=20, r_seed=79, folds=10)
#plot_kfold_figs(MSE_train, MSE_test, polydegrees, lambdas_)

#if __name__ == '__main__':
    #n_points = 20 
    #noisy = True
    #x,y = generate_determ_data(n_points)
    #n_lambdas = 6

    # Lambda values used for bootstrap resampling
    #bootstrap_lambdas_ = np.logspace(-6, -2, n_lambdas)

    # These list of lambda variables is used in kFold cross-validation
    #kFold_lambdas_ = np.logspace(-6, -2, n_lambdas)

    #possibly acceptable results
    #MSE_train, MSE_test, bias, variance, polydegrees, = Lasso_reg_boot(x,y,lambdas_=bootstrap_lambdas_,degrees=11, n_points=10, r_seed=79,noisy=noisy, centering=True)
    #plot_figs_bootstrap_all_lambdas(MSE_train, MSE_test, bias, variance, polydegrees, bootstrap_lambdas_)

    #MSE_train, MSE_test, bias, variance, polydegrees, = Lasso_reg_boot(x,y,lambdas_=bootstrap_lambdas_,degrees=11, n_points=20, r_seed=79, noisy=noisy, centering=True)
    #plot_figs_bootstrap_all_lambdas(MSE_train, MSE_test, bias, variance, polydegrees, bootstrap_lambdas_)

    
    
    #MSE_train, MSE_test, polydegrees, lambdas_ = Lasso_reg_kFold(x,y,lambdas_=kFold_lambdas_, n_points=10, r_seed=79, folds=10, centering=False, noisy=noisy)
    #plot_kfold_figs_for_k(MSE_train, MSE_test, polydegrees, lambdas_)



def task_f(n_points=20, n_lambdas=6, r_seed=79, n_boots=100, degrees=12,       
            noisy=True, centering=True, compare=False, kfold_for_all_lam=False):

    x,y = generate_determ_data(n_points)
    lambdas = np.logspace(-6,-1,n_lambdas)


    MSE_train_boot, MSE_test_boot, bias_, variance_, deg = Lasso_reg_boot(x, y, lambdas_=lambdas, r_seed=r_seed, n_points=n_points,
                                                                n_boots=n_boots, degrees=degrees, centering=centering) 
    lam, index = find_best_lambda(lambdas, MSE_train_boot)

    plot_figs_bootstrap_all_lambdas(MSE_train_boot, MSE_test_boot, variance_, bias_, deg, lambdas)

    folds = [5,8,10] 
    MSE_train_folds_L = np.empty((len(folds), degrees))
    MSE_test_folds_L = np.empty((len(folds), degrees))

    for i in range(len(folds)):
        
        MSE_train, MSE_test, deg = Lasso_reg_kFold(x,y,lambdas_=np.array([lam]), degrees=degrees, folds=folds[i], r_seed=r_seed, centering=centering)
        MSE_train_folds_L[i], MSE_test_folds_L[i] = MSE_train, MSE_test
    

    plot_kFold_figs_for_L(MSE_train_folds_L, MSE_test_folds_L, deg, folds)
    if kfold_for_all_lam: 
        MSE_train_all_folds, MSE_test_all_folds, deg = Lasso_reg_kFold(x,y,lambdas_=lambdas, degrees=degrees, folds=folds[len(folds)-1], r_seed=r_seed, centering=centering)
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

    # good random_seeds = [79, 227"""

task_f(kfold_for_all_lam=True)