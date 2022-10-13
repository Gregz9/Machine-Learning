import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import generate_determ_data
from Regression import OLS_reg_kFold, OLS_reg_kFold_scikit_learn
from plot_functions import plot_kFold_figs_for_L

def task_d(n_points=20, order=10, noisy=True, centering=False, include_comparison=True):  

    x,y = generate_determ_data(n_points)
    folds = [5,6,8,10]
    MSE_train_folds = np.empty((len(folds), order))
    MSE_test_folds = np.empty((len(folds), order))

    if include_comparison:
        MSE_test_scikit = np.empty((len(folds), order))
    else: 
        MSE_test_scikit = []

    for i in range(len(folds)):
        MSE_train, MSE_test, pol = OLS_reg_kFold(x,y,n_points=n_points, noisy=noisy, degrees=order, r_seed=79, folds=folds[i], scaling=centering)
        MSE_train_folds[i], MSE_test_folds[i] = MSE_train, MSE_test
        if include_comparison: 
            _, MSE_t_sci, _ = OLS_reg_kFold_scikit_learn(x,y,n_points=n_points, noisy=noisy, degrees=order, r_seed=79, folds=folds[i], scaling=True)
            MSE_test_scikit[i] = MSE_t_sci
        
    plot_kFold_figs_for_L(MSE_train_folds, MSE_test_folds, pol, folds, MSE_test_scikit)

task_d()