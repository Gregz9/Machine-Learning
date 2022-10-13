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
    compute_optimal_parameters, predict, R2, MSE, compute_betas_ridge)
from Regression import OLS_reg, Ridge_reg, Lasso_reg
from plot_functions import compare_2_predictions, plot_figs_single_run
    
def task_a(n_points=20, noisy=True, centering=True, degrees=11, r_seed=79):

    x, y = generate_determ_data(n_points)

    (betas, MSE_train, MSE_test, 
    R2_train, R2_test, preds_cn, x, y, z) = OLS_reg(x, y, n_points=n_points, scaling=centering, noisy=noisy, degrees=degrees, r_seed=79)

    plot_figs_single_run(MSE_train, MSE_test, R2_train, R2_test, betas, reg_type='ols')
    compare_2_predictions(x,y,z,preds_cn[4], 5)

task_a()