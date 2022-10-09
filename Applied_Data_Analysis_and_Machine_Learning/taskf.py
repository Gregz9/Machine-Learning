from distutils.command.build_scripts import first_line_re
from random import random
import numpy as np
from Regression import Reg_model 
from sklearn.model_selection import train_test_split
from utils import ( 
    FrankeFunction, generate_determ_data, create_X, create_simple_X,
    KFold_split, generate_design_matrix, predict, compute_betas_ridge, MSE,
    compute_optimal_parameters)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

def plot_figs(*args):
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

    axs[0,0].plot(args[4], args[0][1], 'b', label='MSE_train') 
    axs[0,0].plot(args[4], args[1][1], 'r', label='MSE_test')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Mean Squared Error')
    axs[0,0].legend()
    axs[0,1].plot(args[4], args[1][1], 'b', label='MSE_test')
    axs[0,1].plot(args[4], args[2][1], 'y', label='bias')
    axs[0,1].plot(args[4], args[3][1], 'g', label='variance')
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
    axs[1,1].legend()

    plt.show()

def Lasso_reg(n_points=20, degrees=10, n_boots=100, n_lambdas=6, sacaling=False, noisy=True, r_seed=79):
    np.random.seed(r_seed) 
    lamb_= np.logspace(-15, -10, n_lambdas)

    lasso = Reg_model(lambas_=lamb_, n_points=n_points, reg_type='lasso', degree=degrees)

    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias_ = np.zeros((n_lambdas, degrees))
    variance_ = np.zeros((n_lambdas, degrees))

    for k in range(len(lasso.lambdas_)): 
        i, i2 = 3, 3
        print(f'Lamda value:{lasso.lambdas_[k]}')
        
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        bias = np.zeros(degrees)
        variance = np.zeros(degrees)
        polydegree = np.zeros(degrees)
        
        for degree in range(1, degrees+1): 
            #print(f'Polynomial degree {degree}')
            X_train, X_test, z_train, z_test = train_test_split(lasso.X[:, :i], lasso.z.ravel(), test_size=0.2)
            lasso.fit_data(X_train, z_train, lambda_=lasso.lambdas_[k])
            z_pred_train = lasso.predict(X_train)
            z_pred_test = lasso.predict(X_test)

            MSE_train_list[degree-1] = MSE(z_train, z_pred_train)
            MSE_test_list[degree-1] = MSE(z_test, z_pred_test)
            polydegree[degree-1] = degree
            i += i2
            i2 += 1 
        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list
    return MSE_train, MSE_test, polydegreoe

#MSE_train, MSE_test, degs = Lasso_reg()


def Lasso_reg_bootstrap(n_points=20, degrees=11, n_boots=100, n_lambdas=6, sacaling=False, noisy=True, r_seed=227):
    np.random.seed(r_seed) 
    x,y = generate_determ_data(size=n_points)
    z = FrankeFunction(x,y, noise=True)
    X=create_X(x,y,degrees)
    X = np.delete(X,0,axis=1)
    lamb_= np.logspace(-2,3, n_lambdas)
    
    scaler = StandardScaler()
    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias_ = np.zeros((n_lambdas, degrees))
    variance_ = np.zeros((n_lambdas, degrees))

    for k in range(len(lamb_)): 
        i, i2 = 3, 3
        print(f'Lamda value:{lamb_[k]}')
        
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        bias = np.zeros(degrees)
        variance = np.zeros(degrees)
        polydegree = np.zeros(degrees)
        lasso = Lasso(lamb_[k], fit_intercept=True )#, precompute=True)
        #model = make_pipeline(PolynomialFeatures(degree=degrees), Lasso(lamb_[k], fit_intercept=False, precompute=True))
        
        for degree in range(1, degrees+1): 
            #model = make_pipeline(PolynomialFeatures(degree=degrees), Lasso(lamb_[k], fit_intercept=False))
        
            #lasso = Lasso(lamb_[k], fit_intercept=False, precompute=True)
            #print(f'Polynomial degree {degree}')
            X_train, X_test, z_train, z_test = train_test_split(X[:,:i], z.ravel(), test_size=0.2)
            x_train_mean = np.mean(X_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)

            X_train_centered = X_train - x_train_mean
            z_train_centered = z_train - z_train_mean
            X_test_centered = X_test - x_train_mean 

            pred_train_avg = np.empty((n_boots, z_train.shape[0]))
            pred_test_avg = np.empty((n_boots, z_test.shape[0]))
            training_error = 0 
            test_error = 0

            for boot in range(n_boots):
                print(boot)

                X_, z_ = resample(X_train_centered, z_train_centered, replace=True)    
            
                lasso.fit(X_, z_)
                z_pred_train = lasso.predict(X_train_centered) + z_train_mean
                z_pred_test = lasso.predict(X_test_centered) + z_train_mean
                #z_pred_test = lasso.predict(X_test)

                pred_train_avg[boot, :] = z_pred_train
                pred_test_avg[boot, :] = z_pred_test
                training_error += MSE(z_train, z_pred_train)
                test_error += MSE(z_test, z_pred_test)
                
            i += i2
            i2 += 1 

            MSE_train_list[degree-1] = training_error/n_boots #np.mean(np.mean((z_train-pred_train_avg)**2, axis=0, keepdims=True))#training_error/n_boots
            MSE_test_list[degree-1] = test_error/n_boots#np.mean(np.mean((z_test-pred_test_avg)**2, axis=0, keepdims=True))
            bias[degree-1] = np.mean((z_test - np.mean(pred_test_avg, axis=0, keepdims=True))**2)
            variance[degree-1] = np.mean(np.var(pred_test_avg, axis=0, keepdims=True))  
           
            polydegree[degree-1] = degree
            
        MSE_train[k] = MSE_train_list
        MSE_test[k] = MSE_test_list
        bias_[k] = bias
        variance_[k] = variance
  
    return MSE_train, MSE_test, bias_, variance_, polydegree, lamb_

#MSE_train, MSE_test, degs = Lasso_reg()
MSE_train, MSE_test, bias_, var_, degs, lambdas = Lasso_reg_bootstrap(n_points=20, r_seed=79, degrees=11, n_boots=100)

fig, axs = plt.subplots(2,3)
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
fig.suptitle('Lasso for varying lambda values')
axs[1,2].plot(degs, MSE_train[5], 'k', label='MSE_train') 
axs[1,2].plot(degs, MSE_test[5], 'r', label='MSE_test')
axs[1,2].set_xlabel('Polynomial order')
axs[1,2].set_ylabel('Mean Squared Error')
axs[1,2].set_title(f'Lambda value{lambdas[5]}')
axs[1,2].legend()

axs[1,1].plot(degs, MSE_train[4], 'b', label='MSE_train') 
axs[1,1].plot(degs, MSE_test[4], 'r', label='MSE_test')
axs[1,1].set_xlabel('Polynomial order')
axs[1,1].set_ylabel('Mean Squared Error')
axs[1,1].set_title(f'Lambda value{lambdas[4]}')
axs[1,1].legend()

axs[1,0].plot(degs, MSE_train[3], 'b', label='MSE_train') 
axs[1,0].plot(degs, MSE_test[3], 'r', label='MSE_test')
axs[1,0].set_xlabel('Polynomial order')
axs[1,0].set_ylabel('Mean Squared Error')
axs[1,0].set_title(f'Lambda value{lambdas[3]}')
axs[1,0].legend()

axs[0,2].plot(degs, MSE_train[2], 'b', label='MSE_train') 
axs[0,2].plot(degs, MSE_test[2], 'r', label='MSE_test')
axs[0,2].set_xlabel('Polynomial order')
axs[0,2].set_ylabel('Mean Squared Error')
axs[0,2].set_title(f'Lambda value{lambdas[2]}')
axs[0,2].legend()

axs[0,1].plot(degs, MSE_train[1], 'b', label='MSE_train') 
axs[0,1].plot(degs, MSE_test[1], 'r', label='MSE_test')
axs[0,1].set_xlabel('Polynomial order')
axs[0,1].set_ylabel('Mean Squared Error')
axs[0,1].set_title(f'Lambda value{lambdas[1]}')
axs[0,1].legend()

axs[0,0].plot(degs, MSE_train[0], 'b', label='MSE_train') 
axs[0,0].plot(degs, MSE_test[0], 'r', label='MSE_test')
axs[0,0].set_xlabel('Polynomial order')
axs[0,0].set_ylabel('Mean Squared Error')
axs[0,0].set_title(f'Lambda value{lambdas[0]}')
axs[0,0].legend()

#axs[1,2].plot(degs, MSE_test[5], 'b', label='MSE_test')
axs[1,2].plot(degs, bias_[5], 'y', label='bias')
axs[1,2].plot(degs, var_[5], 'g', label='variance')
axs[1,2].set_xlabel('Polynomial order')
axs[1,2].set_ylabel('Mean Squared Error')
axs[0,1].legend()
plt.show()
#plot_figs(MSE_train, MSE_test, bias_, var_, degs)
#color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
