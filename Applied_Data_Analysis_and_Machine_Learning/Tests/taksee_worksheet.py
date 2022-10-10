from ensurepip import bootstrap
from statistics import variance
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
import sklearn
from sklearn.pipeline import make_pipeline
from numpy.core import _asarray
from sklearn.utils import resample
import time
from utils import ( 
    FrankeFunction, generate_determ_data, create_X, create_simple_X,
    KFold_split, generate_design_matrix, predict, compute_betas_ridge, MSE,
    compute_optimal_parameters, compute_lasso_parameteres)

def plot_figs(*args):
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0,0].plot(args[4][0], args[0][0], 'b', label='MSE_train') 
    axs[0,0].plot(args[4][0], args[1][0], 'r', label='MSE_test')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Mean Squared Error')
    axs[0,0].legend()
    """axs[0,1].plot(args[4], args[1][5], 'b', label='MSE_test')
    axs[0,1].plot(args[4], args[2][5], 'y', label='bias')
    axs[0,1].plot(args[4], args[3][5], 'g', label='variance')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('Mean Squared Error')
    axs[0,1].legend()

    axs[1,0].plot(args[4], args[0][0], 'b', label='MSE_train') 
    axs[1,0].plot(args[4], args[1][0], 'r', label='MSE_test')
    axs[1,0].set_xlabel('Polynomial order')
    axs[1,0].set_ylabel('Mean Squared Error')
    axs[1,0].legend()

    axs[1,1].plot(args[4], args[1][0], 'b', label='MSE_test')
    axs[1,1].plot(args[4], args[2][0], 'y', label='bias')
    axs[1,1].plot(args[4], args[3][0], 'g', label='variance')
    axs[1,1].set_xlabel('Polynomial order')
    axs[1,1].set_ylabel('Mean Squared Error')
    axs[1,1].legend()"""

    plt.show()

def Ridge_reg_bootstrap(n_points=20, degrees=10, n_boots=100, n_lambdas=6, scaling=False, noisy=True, r_seed=7): 
    np.random.seed(r_seed)
    x,y = generate_determ_data(n_points)
    z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees)
    lambdas = np.logspace(-12,-3,n_lambdas)
    print(lambdas)

    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias_ = np.zeros((n_lambdas, degrees))
    variance_ = np.zeros((n_lambdas, degrees))
    #polydegree = np.zeros(degrees)

    
    start_time = time.time()
    for k in range(len(lambdas)): 
        #print(f'Lamda value:{lambda_}')
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        bias = np.zeros(degrees)
        variance = np.zeros(degrees)
        polydegree = np.zeros(degrees)
        i, i2 = 3, 3
        for degree in range(1, degrees+1): 
            #print(f'Polynomial degree {degree}')
            I = np.eye(i,i)
            X_train, X_test, z_train, z_test = train_test_split(X[:, :i], z.ravel(), test_size=0.2)
            pred_train_avg = np.empty((n_boots, z_train.shape[0]))
            pred_test_avg = np.empty((n_boots, z_test.shape[0]))
            test_error = 0
            training_error = 0 
            for boot in range(n_boots):

                X_, z_ = resample(X_train, z_train, replace=True)
               
                betas_ridge  = compute_lasso_parameteres(X_, z_, lambdas[k])

                z_pred_train = predict(X_train, betas_ridge)
                z_pred_test = predict(X_test, betas_ridge)

                pred_train_avg[boot, :] = z_pred_train
                pred_test_avg[boot, :] = z_pred_test 
                #training_error += MSE(z_train, z_pred_train)
                #test_error += MSE(z_test, z_pred_test)
            
            i += i2
            i2 += 1 
            MSE_train_list[degree-1] = np.mean(np.mean((z_train-pred_train_avg)**2, axis=0, keepdims=True))#training_error/n_boots
            MSE_test_list[degree-1] = np.mean(np.mean((z_test-pred_test_avg)**2, axis=0, keepdims=True))#test_error/n_boots
            polydegree[degree-1] = degree   
            bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
            variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))  
        
        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list
        bias_[k] = bias
        variance_[k] = variance

    #print(X_test)
    print("Time used %s seconds" % (time.time() - start_time))
    return MSE_train, MSE_test, bias_, variance_, polydegree
MSE_train, MSE_test, deg, bias_, variance_ = Ridge_reg_bootstrap(r_seed=427, n_points=10, n_boots=100, degrees=10)
print(MSE_train[0].shape)
print(MSE_test[0].shape)
print(deg[0].shape)
plot_figs(MSE_train, MSE_test, bias_, variance_, deg)

def Ridge_reg_Kfold(n_points=20, degrees=10, folds=5, n_lambdas=6, scaling=False, noisy=True, r_seed=427, include_wrong_calc=False):
    np.random.seed(r_seed)
    x,y = generate_determ_data(n_points)
    z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees)
    lambdas = np.logspace(-12,-3,n_lambdas)
    z=z.ravel()
    #train_ind, test_ind = KFold_split(z=z, k=folds)
    kfold = KFold(n_splits=folds)

    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias_ = np.zeros((n_lambdas, degrees))
    variance_ = np.zeros((n_lambdas, degrees))
    polydegree = np.zeros(degrees)

    for k in range(len(lambdas)): 
        #print(f'Lamda value:{lambda_}')
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        bias = np.zeros(degrees)
        variance = np.zeros(degrees)
       
        i, i2 = 3, 3
        for degree in range(1, degrees+1): 
            #print(f'Polynomial degree {degree}')
            I = np.eye(i,i)
            pred_train_avg = []
            pred_test_avg = []
            z_train_set = []
            z_test_set = []
            training_error = 0 
            test_error = 0 

            for train_indx, test_indx in kfold.split(X):#zip(train_ind, test_ind):
                
                x_train, z_train = X[train_indx, :i], z[train_indx]
                x_test, z_test = X[test_indx, :i], z[test_indx]
                z_train_set.append(z_train)
                z_test_set.append(z_test)

                betas = compute_betas_ridge(x_train, z_train, lambdas[k]*I)
                
                z_pred_train = predict(x_train, betas)
                z_pred_test = predict(x_test, betas)
                pred_train_avg.append(z_pred_train)
                pred_test_avg.append(z_pred_test)
            
            i += i2 
            i2 += 1

            if (n_points**2)%folds > 0:
                testInd = min([test.shape for test in pred_test_avg])
                testInd = [i for i in range(len(pred_test_avg)) if pred_test_avg[i].shape == testInd][0]
                tst = np.concatenate((z_test_set[:testInd] + z_test_set[testInd+1:]))

                trainInd = min([train.shape for train in pred_train_avg])
                trainInd = [i for i in range(len(pred_train_avg)) if pred_train_avg[i].shape == trainInd][0]
                trn = np.concatenate((z_train_set[:trainInd] + z_train_set[trainInd+1:]))

                tst_pred = np.concatenate((pred_test_avg[:testInd] + pred_test_avg[testInd+1:]))
                trn_pred = np.concatenate((pred_train_avg[:trainInd] + pred_train_avg[trainInd+1:]))

                MSE_train_list[degree-1] =  np.mean(np.mean((trn-trn_pred)**2, axis=0, keepdims=True))  #training_error/folds
                MSE_test_list[degree-1] = np.mean(np.mean((tst-tst_pred)**2, axis=0, keepdims=True)) #test_error/folds
          
                bias[degree-1] =  np.mean((tst - np.mean(tst_pred, keepdims=True))**2)
                variance[degree-1] = np.mean(np.var(tst_pred, keepdims=True))
                polydegree[degree-1] = degree
            else: 
                MSE_train_list[degree-1] = np.mean(np.mean((np.array(z_train_set)-np.array(pred_train_avg))**2, axis=0, keepdims=True))#training_error/folds#
                MSE_test_list[degree-1] = np.mean(np.mean((np.array(z_test_set)-np.array(pred_test_avg))**2, axis=0, keepdims=True)) #+ np.mean(np.mean((z_test_set[testInd]-pred_test_avg[testInd])**2, axis=0, keepdims=True))#test_error/n_bootstest_error/folds#
                bias[degree-1] =  np.mean((z_test_set - np.mean(pred_test_avg, keepdims=True))**2) 
                variance[degree-1] = np.mean(np.var(pred_test_avg, keepdims=True)) 
                polydegree[degree-1] = degree

        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list
        bias_[k] = bias
        variance_[k] = variance

    return MSE_train, MSE_test, bias_, variance_, polydegree

#MSE_train, MSE_test, bias, var, deg = Ridge_reg_Kfold(folds=9, r_seed=427)
#plot_figs(MSE_train, MSE_test, bias, var, deg)