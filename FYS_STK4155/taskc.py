import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
import sklearn
from numpy.core import _asarray
from sklearn.utils import resample
from utils import ( 
    FrankeFunction, generate_data, generate_design_matrix, 
    compute_optimal_parameters, predict, MSE)

def OLS_boot_reg(n_points=40, degrees=5, n_boots=10, seed=None): 

    x, y = generate_data(n_points, seed)
    z = FrankeFunction(x,y)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    betas_list = np.empty(degrees)
    preds_cn = np.empty(degrees)

    for degree in range(1, degrees+1): 
            
        X = generate_design_matrix(x, y, degree, intercept=False)
        MSE_train_avg = np.empty((n_boots))
        MSE_test_avg = np.empty((n_boots))
        betas_avg = np.empty((n_boots))
        preds_avg = np.empty((n_boots))

        X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2, random_state=seed)
        z_pred_train = np.empty((z_train.shape[0], n_boots))
        z_pred_test = ((z_test.shape[0], n_boots))
        for j in range(n_boots): 
            # Bootstrap resampling of datasets after split
            X_train, z_train = resample(X_train, z_train)
            X_test, z_test = resample(X_test, z_test)

            #Centering datasets
            x_train_mean = np.mean(X_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)     

            # Using centered values of X and y to compute parameters beta
            X_train_centered = X_train - x_train_mean
            z_train_centered = z_train - z_train_mean
            X_test_centered = X_test - x_train_mean 

            beta_SVD_cn = compute_optimal_parameters(X_train_centered, z_train_centered)
            betas_avg[j] = (beta_SVD_cn)

            intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)

            preds_visualization_cn = predict(X, beta_SVD_cn, z_train_mean)
            preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
            preds_cn.append(preds_visualization_cn)

            z_pred_train[:, j] = predict(X_train_centered, beta_SVD_cn, z_train_mean) 
            z_pred_test[:, j] = predict(X_test_centered, beta_SVD_cn, z_train_mean)

            MSE_train_avg[j] = MSE(z_train, z_pred_train[:, j])
            MSE_test_avg[j] = MSE(z_test, z_pred_test[:, j])

        betas_list[degree] = np.mean(betas_avg, axis=0) 
        MSE_train_list[degree] = np.mean(())
        MSE_test_list.append(MSE(z_test, z_pred_test[:, j]))


    return betas_list, preds_cn, MSE_train_list, MSE_test_list