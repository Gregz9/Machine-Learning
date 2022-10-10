from operator import ge
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from utils import (FrankeFunction, generate_determ_data, create_X, MSE)
import cvxpy as cp

def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def lasso_reg_boot(n_points=20, degrees=10, n_boots=100, n_lambdas=6, noisy=True, r_seed=79): 

    np.random.seed(r_seed)
    x,y = generate_determ_data(n_points)
    z =FrankeFunction(x,y, noise=True)
    X = create_X(x,y, degrees)
    lambdas_ = np.logspace(-3, -1.5, n_lambdas)

    MSE_train = np.empty((n_lambdas, degrees))
    MSE_test = np.empty((n_lambdas, degrees))
    bias = np.empty((n_lambdas, degrees))
    variance = np.empty((n_lambdas, degrees))

    for f in range(len(lambdas_)): 
        print(f'Level {f}')
        i, i2 = 3,3
        MSE_train_list = np.empty(degrees)
        MSE_test_list = np.empty(degrees)
        bias_ = np.empty(degrees)
        variance_ = np.empty(degrees)
        polydegrees = np.empty(degrees)
        #lasso = Lasso(lambdas_[f], fit_intercept=False, max_iter=200)
        

        for degree in range(1, degrees+1):
            print(f'Degree {degree}')
            x_train, x_test, z_train, z_test = train_test_split(X[:,:i], z.ravel(), test_size=0.2)
            num_betas = x_train.shape[1]
            beta = cp.Variable(num_betas)
            lambd = cp.Parameter(nonneg=True)
            #pred_train = np.empty((n_boots, z_train.shape[0]))
            #pred_test = np.empty((n_boots, z_test.shape[0]))
            training_error = 0
            test_error = 0 

            for boot in range(n_boots):
                x_, z_ = resample(x_train, z_train)
                #for r in range(x_.shape[1]):
                problem = cp.Problem(cp.Minimize(objective_fn(x_, z_, beta, lambd)))
                lambd.value = lambdas_[f]
                problem.solve()
                training_error += mse(x_train, z_train, beta)
                test_error += mse(x_test, z_test, beta)

                #lasso.fit(x_, z_)

                #pred_train[boot, :] = lasso.predict(x_train)
                #pred_test[boot, :] = lasso.predict(x_test)

            i+= i2
            i2 += 1
            MSE_train_list[degree-1] = training_error/n_boots #np.mean(np.mean((z_train-pred_train)**2, axis=0, keepdims=True))
            MSE_test_list[degree-1] = test_error/n_boots#np.mean(np.mean((z_test-pred_test)**2, axis=0, keepdims=True))
            #bias_[degree-1] = np.mean((z_test - np.mean(pred_test, axis=0, keepdims=True))**2)
            #variance_[degree-1] = np.mean(np.var(pred_test, axis=0, keepdims=True))
            polydegrees[degree-1] = degree
        
        MSE_train[f] = MSE_train_list
        MSE_test[f] = MSE_test_list
        bias[f] = bias_
        variance[f] = variance_
    return MSE_train, MSE_test, bias, variance, polydegrees, lambdas_


MSE_train, MSE_test, bias, variance, polydegrees, lambdas_ = lasso_reg_boot(n_points=20, r_seed=79, n_boots=100)

fig, axs = plt.subplots(2,3)
fig.suptitle('MSE of lasso regression for varying values of lambda parameter')

axs[0,0].plot(polydegrees, MSE_train[0], 'g', label='MSE_train')
axs[0,0].plot(polydegrees, MSE_test[0], 'b', label='MSE_test')
axs[0,0].plot(polydegrees, bias[0], 'y', label='bias')
axs[0,0].plot(polydegrees, variance[0], 'r', label='variance')
axs[0,0].set_xlabel('Polynomial_order')
axs[0,0].set_ylabel('MSE')
axs[0,0].set_title(f'Lambda: {lambdas_[0]}')
axs[0,0].legend()

axs[0,1].plot(polydegrees, MSE_train[1], 'g', label='MSE_train')
axs[0,1].plot(polydegrees, MSE_test[1], 'b', label='MSE_test')
axs[0,1].plot(polydegrees, bias[1], 'y', label='bias')
axs[0,1].plot(polydegrees, variance[1], 'r', label='variance')
axs[0,1].set_xlabel('Polynomial_order')
axs[0,1].set_ylabel('MSE')
axs[0,1].set_title(f'Lambda: {lambdas_[1]}')
axs[0,1].legend()

axs[0,2].plot(polydegrees, MSE_train[2], 'g', label='MSE_train')
axs[0,2].plot(polydegrees, MSE_test[2], 'b', label='MSE_test')
axs[0,2].plot(polydegrees, bias[2], 'y', label='bias')
axs[0,2].plot(polydegrees, variance[2], 'r', label='variance')
axs[0,2].set_xlabel('Polynomial_order')
axs[0,2].set_ylabel('MSE')
axs[0,2].set_title(f'Lambda: {lambdas_[2]}')
axs[0,2].legend()

axs[1,0].plot(polydegrees, MSE_train[3], 'g', label='MSE_train')
axs[1,0].plot(polydegrees, MSE_test[3], 'b', label='MSE_test')
axs[1,0].plot(polydegrees, bias[3], 'y', label='bias')
axs[1,0].plot(polydegrees, variance[3], 'r', label='variance')
axs[1,0].set_xlabel('Polynomial_order')
axs[1,0].set_ylabel('MSE')
axs[1,0].set_title(f'Lambda: {lambdas_[3]}')
axs[1,0].legend()

axs[1,1].plot(polydegrees, MSE_train[4], 'g', label='MSE_train')
axs[1,1].plot(polydegrees, MSE_test[4], 'b', label='MSE_test')
axs[1,1].plot(polydegrees, bias[4], 'y', label='bias')
axs[1,1].plot(polydegrees, variance[4], 'r', label='variance')
axs[1,1].set_xlabel('Polynomial_order')
axs[1,1].set_ylabel('MSE')
axs[1,1].set_title(f'Lambda: {lambdas_[4]}')
axs[1,1].legend()

axs[1,2].plot(polydegrees, MSE_train[5], 'g', label='MSE_train')
axs[1,2].plot(polydegrees, MSE_test[5], 'b', label='MSE_test')
axs[1,2].plot(polydegrees, bias[5], 'y', label='bias')
axs[1,2].plot(polydegrees, variance[5], 'r', label='variance')
axs[1,2].set_xlabel('Polynomial_order')
axs[1,2].set_ylabel('MSE')
axs[1,2].set_title(f'Lambda: {lambdas_[5]}')
axs[1,2].legend()
plt.show()
