
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (generate_determ_data, create_X, FrankeFunction, MSE, KFold_split,
                    compute_betas_ridge, predict, R2, compute_optimal_parameters2) 
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso

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
            

def OLS_reg_kFold(x,y,z=None,n_points=20, degrees=5, folds=5, scaling=False, noisy=True, r_seed=79): 
    np.random.seed(r_seed)
    
    if z == None: 
        z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    z = z.ravel()

    MSE_train = np.empty(degrees)
    MSE_test = np.empty(degrees)

    polydegree = np.zeros(degrees)

    for degree in range(1, degrees+1): 
        pred_train_avg = []
        pred_test_avg = []
        training_error = 0 
        test_error = 0
        print(f'Polynomial degree {degree}')
        train_ind, test_ind = KFold_split(z=z, k=folds)
        for train_indx, test_indx in zip(train_ind, test_ind):
            
            x_train, z_train = X[train_indx, :int((degree+1)*(degree+2)/2)], z[train_indx]
            x_test, z_test = X[test_indx, :int((degree+1)*(degree+2)/2)], z[test_indx]
            if scaling:
                x_train_mean = np.mean(x_train, axis=0) 
                z_train_mean = np.mean(z_train, axis=0)  
                x_train -= x_train_mean
                x_test -= x_train_mean
                z_train_centered = z_train - z_train_mean
            else: 
                z_train_centered = z_train
                z_train_mean = 0 
            
            betas = compute_optimal_parameters2(x_train, z_train_centered)
            z_pred_train = predict(x_train, betas, z_train_mean)
            z_pred_test = predict(x_test, betas, z_train_mean)

            pred_train_avg.append(z_pred_train)
            pred_test_avg.append(z_pred_test)
            training_error += MSE(z_train, z_pred_train)
            test_error += MSE(z_test, z_pred_test)

        MSE_train[degree-1] = training_error/folds 
        MSE_test[degree-1] = test_error/folds 
        polydegree[degree-1] = degree

    return MSE_train, MSE_test, polydegree

def OLS_reg_kFold_scikit_learn(x,y,z=None,n_points=20, degrees=5, folds=5, scaling=False, noisy=True, r_seed=79): 
    np.random.seed(r_seed)
    
    if z == None: 
        z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    z = z.ravel()

    MSE_train = np.empty(degrees)
    MSE_test = np.empty(degrees)

    polydegree = np.zeros(degrees)

    for degree in range(1, degrees+1): 
        pred_train_avg = []
        pred_test_avg = []
        training_error = 0 
        test_error = 0
        print(f'Polynomial degree {degree}')
        kFold = KFold(n_splits=folds)
        for train_indx, test_indx in kFold.split(X):
            
            x_train, z_train = X[train_indx, :int((degree+1)*(degree+2)/2)], z[train_indx]
            x_test, z_test = X[test_indx, :int((degree+1)*(degree+2)/2)], z[test_indx]
            if scaling:
                x_train_mean = np.mean(x_train, axis=0) 
                z_train_mean = np.mean(z_train, axis=0)  
                x_train -= x_train_mean
                x_test -= x_train_mean
                z_train_centered = z_train - z_train_mean
            else: 
                z_train_centered = z_train
                z_train_mean = 0 
            
            betas = compute_optimal_parameters2(x_train, z_train_centered)
            z_pred_train = predict(x_train, betas, z_train_mean)
            z_pred_test = predict(x_test, betas, z_train_mean)

            pred_train_avg.append(z_pred_train)
            pred_test_avg.append(z_pred_test)
            training_error += MSE(z_train, z_pred_train)
            test_error += MSE(z_test, z_pred_test)

        MSE_train[degree-1] = training_error/folds 
        MSE_test[degree-1] = test_error/folds 
        polydegree[degree-1] = degree

    return MSE_train, MSE_test, polydegree


def Ridge_reg_kFold(x, y, lambdas, z =None, n_points=20, degrees=10, folds=5, scaling=False, noisy=True, r_seed=79):
    np.random.seed(r_seed)
    if z == None: 
        z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    z=z.ravel()
    train_ind, test_ind = KFold_split(z=z, k=folds)
    
    MSE_train = np.empty((lambdas.shape[0], degrees))
    MSE_test = np.empty((lambdas.shape[0], degrees))
    

    for k in range(len(lambdas)): 
        print(f'Lamda value:{lambdas[k]}')
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        polydegree = np.zeros(degrees)
        for degree in range(1, degrees+1): 
            print(f'Polynomial degree {degree}')

            training_error = 0 
            test_error = 0 

            for train_indx, test_indx in zip(train_ind, test_ind):
                
                x_train, z_train = X[train_indx, :int((degree+1)*(degree+2)/2)], z[train_indx]
                x_test, z_test = X[test_indx, :int((degree+1)*(degree+2)/2)], z[test_indx]
                if scaling:
                    x_train_mean = np.mean(x_train, axis=0) 
                    z_train_mean = np.mean(z_train, axis=0)  
                    x_train -= x_train_mean
                    x_test -= x_train_mean
                    z_train_centered = z_train - z_train_mean
                else: 
                    z_train_centered = z_train
                    z_train_mean = 0 

                betas = compute_betas_ridge(x_train, z_train_centered, lambdas[k])
                
                z_pred_train = predict(x_train, betas, z_train_mean)
                z_pred_test = predict(x_test, betas, z_train_mean)
                training_error += MSE(z_train, z_pred_train)
                test_error += MSE(z_test, z_pred_test)

            MSE_train_list[degree-1] = training_error/folds
            MSE_test_list[degree-1] = test_error/folds 
            polydegree[degree-1] = degree

        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list

    return MSE_train, MSE_test, polydegree


def Ridge_reg_boot(x, y, lambdas, z=None, n_points=20, degrees=10, n_boots=100, n_lambdas=6, scaling=False, noisy=True, r_seed=79): 
    np.random.seed(r_seed)
    if z == None:
        z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees, centering=scaling)
    
    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias_ = np.zeros((n_lambdas, degrees))
    variance_ = np.zeros((n_lambdas, degrees))

    for k in range(len(lambdas)): 
        print(f'Lamda value:{lambdas[k]}')
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

            for boot in range(n_boots):

                X_, z_ = resample(x_train, z_train_centered)
                betas_ridge  = compute_betas_ridge(X_, z_, lambdas[k])

                z_pred_train = predict(x_train, betas_ridge, z_train_mean)
                z_pred_test = predict(x_test, betas_ridge, z_train_mean)

                pred_train_avg[boot, :] = z_pred_train
                pred_test_avg[boot, :] = z_pred_test 

            MSE_train_list[degree-1] = np.mean(np.mean((z_train-pred_train_avg)**2, axis=0, keepdims=True))#training_error/n_boots
            MSE_test_list[degree-1] = np.mean(np.mean((z_test-pred_test_avg)**2, axis=0, keepdims=True))#test_error/n_boots
            bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
            variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))  
            polydegree[degree-1] = degree   
        
        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list
        bias_[k] = bias
        variance_[k] = variance

    return MSE_train, MSE_test, bias_, variance_, polydegree


def Lasso_reg_boot(x, y, lambdas_, z=None, n_points=20, degrees=10, n_boots=100, n_lambdas=6, noisy=True, centering=False, r_seed=79): 

    np.random.seed(r_seed)
    if z==None: 
        z =FrankeFunction(x,y, noise=noisy)
    X = create_X(x,y, degrees, centering=centering)
    
    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias = np.empty((n_lambdas, degrees))
    variance = np.empty((n_lambdas, degrees))

    for f in range(len(lambdas_)): 
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        bias_ = np.empty(degrees)
        variance_ = np.empty(degrees)
        polydegrees = np.empty(degrees)

        for degree in range(1, degrees+1):
            print(f'Processing polynomial of {degree} degree ')
            x_train, x_test, z_train, z_test = train_test_split(X[:,:int((degree+1)*(degree+2)/2)], z.ravel(), test_size=0.2)
            if centering: 
                #lasso = Lasso(lambdas_[f], fit_intercept=True, max_iter=1e4, tol=1e-2)
                lasso = Lasso(lambdas_[f], fit_intercept=True, max_iter=300, tol=1e-2)
                x_train_mean = np.mean(x_train, axis=0) 
                z_train_mean = np.mean(z_train, axis=0)  
                x_train -= x_train_mean
                x_test -= x_train_mean
                z_train_centered = z_train - z_train_mean
            else: 
                #lasso = Lasso(lambdas_[f], fit_intercept=False, max_iter=1e4, tol=1e-2)
                lasso = Lasso(lambdas_[f], fit_intercept=True, max_iter=300, tol=1e-2)
                z_train_mean = 0
                z_train_centered = z_train

            pred_train = np.empty((n_boots, z_train.shape[0]))
            pred_test = np.empty((n_boots, z_test.shape[0]))

            for boot in range(n_boots):
                x_, z_ = resample(x_train, z_train_centered)
                lasso.fit(x_, z_)

                pred_train[boot, :] = lasso.predict(x_train) + z_train_mean
                pred_test[boot, :] = lasso.predict(x_test) + z_train_mean

 
            MSE_train_list[degree-1] = np.mean(np.mean((z_train-pred_train)**2, axis=0, keepdims=True))
            MSE_test_list[degree-1] = np.mean(np.mean((z_test-pred_test)**2, axis=0, keepdims=True))
            bias_[degree-1] = np.mean((z_test - np.mean(pred_test, axis=0, keepdims=True))**2)
            variance_[degree-1] = np.mean(np.var(pred_test, axis=0, keepdims=True))
            polydegrees[degree-1] = degree
        
        MSE_train[f] = MSE_train_list
        MSE_test[f] = MSE_test_list
        bias[f] = bias_
        variance[f] = variance_
    return MSE_train, MSE_test, bias, variance, polydegrees


def Lasso_reg_kFold(x,y,lambdas_,z=None,n_points=20, degrees=10, folds=5, noisy=True, r_seed=79, centering=False): 

    np.random.seed(r_seed)
    if z==None: 
        z =FrankeFunction(x,y, noise=noisy)
    X = create_X(x,y, degrees, centering=centering)
    z=z.ravel()
    train_ind, test_ind = KFold_split(z=z, k=folds)

    MSE_train = np.empty((lambdas_.shape[0], degrees))
    MSE_test = np.empty((lambdas_.shape[0], degrees))

    for f in range(len(lambdas_)): 
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        polydegrees = np.empty(degrees)
        if centering: 
            lasso = Lasso(lambdas_[f], fit_intercept=True, max_iter=1000, tol =1e-2)
        else: 
            lasso = Lasso(lambdas_[f], fit_intercept=False, max_iter=1000, tol=1e-2)

        for degree in range(1, degrees+1):
           
            training_error = 0 
            test_error = 0 

            for train_indx, test_indx in zip(train_ind, test_ind):

                x_train, z_train = X[train_indx, :int((degree+1)*(degree+2)/2)], z[train_indx]
                x_test, z_test = X[test_indx, :int((degree+1)*(degree+2)/2)], z[test_indx]
                if centering: 
                    x_train_mean = np.mean(x_train, axis=0) 
                    z_train_mean = np.mean(z_train, axis=0)  
                    x_train -= x_train_mean
                    x_test -= x_train_mean
                    z_train_centered = z_train - z_train_mean
                else: 
                    z_train_mean = 0
                    z_train_centered = z_train
            
                lasso.fit(x_train, z_train_centered)

                training_error += MSE(z_train, (lasso.predict(x_train) + z_train_mean) )
                test_error += MSE(z_test, (lasso.predict(x_test) + z_train_mean) )

            MSE_train_list[degree-1] = training_error/folds
            MSE_test_list[degree-1] = test_error/folds 
            polydegrees[degree-1] = degree
        
        MSE_train[f] = MSE_train_list
        MSE_test[f] = MSE_test_list

    return MSE_train, MSE_test, polydegrees, lambdas_