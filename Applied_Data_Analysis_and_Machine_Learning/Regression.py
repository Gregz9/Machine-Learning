
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (generate_determ_data, create_X, FrankeFunction, MSE,
                    compute_betas_ridge, predict, R2, compute_optimal_parameters2) 
from sklearn import linear_model
from sklearn.utils import resample



def OLS_reg(x, y, z=None, n_points=20, degrees=10, r_seed=79, noisy=True, scaling=True): 
    np.random.seed(r_seed)
    
    X = create_X(x,y,degrees, centering=scaling)
    if z == None: 
        z = FrankeFunction(x,y, noise=noisy)
    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    R2_train_list = np.empty(degrees)
    R2_test_list = np.empty(degrees)
    betas_list = []
    preds_cn = []

    for degree in range(1, degrees+1): 
        x_train, x_test, z_train, z_test = train_test_split(X[:, :int((degree+1)*(degree+2)/2)], z.ravel(), test_size=0.2)

        if scaling:
            x_train_mean = np.mean(x_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)  
            x_train -= x_train_mean
            x_test -= x_train_mean
            z_train_centered = z_train - z_train_mean
        else: 
            z_train_mean = 0
            z_train_centered = z_train            

        beta_SVD_cn = compute_optimal_parameters2(x_train, z_train_centered)
        betas_list.append(beta_SVD_cn)
        # Shifted intercept for use when data is not centered
        #intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)
        
        preds_visualization_cn = predict(X[:, :int((degree+1)*(degree+2)/2)], beta_SVD_cn, z_train_mean)
        preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
        preds_cn.append(preds_visualization_cn)

        preds_train_cn = predict(x_train, beta_SVD_cn, z_train_mean)
        preds_test_cn = predict(x_test, beta_SVD_cn, z_train_mean)

        MSE_train_list[degree-1] = MSE(z_train, preds_train_cn)
        MSE_test_list[degree-1] = MSE(z_test, preds_test_cn)
        R2_train_list[degree-1] = R2(z_train, preds_train_cn)
        R2_test_list[degree-1] = R2(z_test, preds_test_cn)

    return betas_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list,  preds_cn, x, y, z


def OLS_reg_boot(x, y,z=None, n_points=20, degrees=5, n_boots=10, scaling=False, noisy=True, r_seed=427): 
    np.random.seed(r_seed)
    if z == None: 
        z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    bias = np.zeros(degrees)
    variance = np.zeros(degrees)
    polydegree = np.zeros(degrees)
    
    for degree in range(1, degrees+1): 
        
        print(f'Processing polynomial of {degree} degree ')
        x_train, x_test, z_train, z_test = train_test_split(X[:, :int((degree+1)*(degree+2)/2)], z.ravel(), test_size=0.2)
        if scaling: 
            x_train_mean = np.mean(x_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)  
            x_train -= x_train_mean
            x_test -= x_train_mean
            z_train_centered = z_train - z_train_mean
        else: 
            z_train_mean = 0
            z_train_centered = z_train

        pred_train_avg = np.empty((n_boots, z_train.shape[0]))
        pred_test_avg = np.empty((n_boots, z_test.shape[0]))

        for j in range(n_boots): 
            
            # Bootstrap resampling of datasets after split
            X_, z_ = resample(x_train, z_train_centered, replace=True)
            beta_SVD = compute_optimal_parameters2(X_, z_)
        
            z_pred_train = predict(x_train, beta_SVD, z_train_mean) 
            z_pred_test = predict(x_test, beta_SVD, z_train_mean)

            pred_train_avg[j, :] = z_pred_train
            pred_test_avg[j, : ] = z_pred_test
        
        MSE_train_list[degree-1] = MSE(z_train, pred_train_avg)#np.mean(np.mean((pred_train_avg-z_train)**2, axis=0, keepdims=True))
        MSE_test_list[degree-1] = MSE(z_test, pred_test_avg) #np.mean(np.mean((pred_test_avg-z_test)**2, axis=0, keepdims=True))
        polydegree[degree-1] = degree
        bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))

    return MSE_train_list, MSE_test_list, bias, variance, polydegree
            


