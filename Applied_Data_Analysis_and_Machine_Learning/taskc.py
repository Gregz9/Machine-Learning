
import numpy as np
from random import random, seed

from utils import generate_determ_data
from Regression import OLS_reg_boot 
from plot_functions import plot_OLS_figs_task_C

def task_c(n_points=10, noisy=True, centering=True,  degrees=11, n_boots=100, r_seed=79):

    x, y = generate_determ_data(n_points)
        
    #bias,var, MSE_train, MSE_test, pol = OLS_boot_reg(n_points=20, degrees=11, n_boots=100, r_seed=79, scaling=True)
    MSE_train, MSE_test, bias, var, pol = OLS_reg_boot(x,y,n_points=n_points, degrees=degrees, n_boots=n_boots, r_seed=r_seed, scaling=centering)
    plot_OLS_figs_task_C(MSE_train, MSE_test, var, bias, pol)

    # Random seeds list when using create_X
    # Good random seeds = [2, 4, 5, 9, 14, 17, 79
    # Good, but some random behavoiur of data sets = [7, 8, 15 
    # Medium random seeds = [10, 12, 1911, 16, 19, 20
    # Weak random seeds = [6, 11, 31, 18

task_c()