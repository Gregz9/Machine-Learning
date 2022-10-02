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
from sklearn.metrics import mean_squared_error

# This file contains all function necessary for running regression

def generate_random_data(size, seed=None): 
# Making data ingestion to the function
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 1, size)) #np.arange(0, 1, 1/size) 
    y = np.sort(np.random.uniform(0, 1, size)) #np.arange(0, 1, 1/size) 
    x, y = np.meshgrid(x,y)
    return x, y

def generate_determ_data(size): 
    x = np.arange(0, 1, 1/size)
    y = np.arange(0, 1, 1/size)
    x, y = np.meshgrid(x,y)
    return x,y

def FrankeFunction(x, y, noise=False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    stoch_noise = np.zeros((len(x),len(x)))
    if noise: 
        stoch_noise = np.random.normal(0, 0.1, len(x)**2)
        stoch_noise = stoch_noise.reshape(len(x), len(x))

    return term1 + term2 + term3 + term4 + stoch_noise

def generate_design_matrix(x, y, order, intercept=True): 
    x = x.ravel() 
    y = y.ravel()
    
    X = np.array((np.ones(len(x))))
    for i in range(order): 
        if i == 0: 
            X = np.column_stack((X, x**(i+1), y**(i+1)))
        else: 
            X = np.column_stack((X, x**(i+1), y**(i+1), (x**i)*(y**i)))

    if not intercept: 
        X = np.delete(X, 0, axis=1)
    return np.array(X) 

def create_simple_X(x, feat): 
    X = np.zeros((len(x), feat))
    for i in range(0, feat): 
        X[:, i] = x**i+x*i
    return X

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.zeros((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X

def compute_optimal_parameters(A, y, intercept=True): 
    # This method uses SVD to compute the optimal parameters beta for OSL.
    # SVD is chosen cause the central matrix of the expression for beta may 
    # cause problems if it's near-singular or singular.
    U, S, VT = np.linalg.svd(A)#, full_matrices=False)
     
    d = np.divide(1.0, S, where=(S!=0))
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)

    beta = (VT.T)@(D.T)@(U.T)@y.ravel()
    return beta

def compute_optimal_parameters2(X, y):
    A = np.linalg.pinv(X.T@X)@X.T
    beta = A@y.ravel()
    return beta

def compute_optimal_parameters_inv(X, y): 
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T)
    beta = beta@(y.ravel())
    return beta

def predict(X, beta, intercept=0): 
    franke_pred = np.array(())
    for i in range(X.shape[0]):
        franke_pred = np.append(franke_pred, np.sum(X[i]*beta) + intercept)

    return franke_pred

def R2(y, y_hat): 
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - np.mean(y_hat))**2)

def MSE(y, y_hat): 
    return np.sum((y-y_hat)**2)/np.size(y_hat)

def split_data(x, y, test_size=0.25, shuffle=False, seed=None): 
    
    y = y.ravel()

    x_train, x_test, y_train, y_test = [], [], [], []

    if not shuffle: 
        x_train = x[ : (round(x.shape[0]*(1-test_size))), : ]
        y_train = y[ : (round(y.shape[0]*(1-test_size)))]
        x_test = x[(round(x.shape[0]*(1-test_size))) : , : ]
        y_test = y[(round(y.shape[0]*(1-test_size))) : ]
    elif shuffle: 
        x,y = sklearn.utils.shuffle(x,y, random_state=seed)
        x_train = x[ : (round(x.shape[0]*(1-test_size))), : ]
        y_train = y[ : (round(y.shape[0]*(1-test_size)))]
        x_test = x[(round(x.shape[0]*(1-test_size))) : , : ]
        y_test = y[(round(y.shape[0]*(1-test_size))) : ]

    return x_train, x_test, y_train, y_test

def KFold_split(z, k): 
    indices = np.random.choice(len(z),len(z), replace=False)
    splits = np.split(indices, k)

    #print(X[splits[0]])
    train_indices = np.empty((k, len(splits[0])*(k-1)))
    test_indices = np.empty((k, len(splits[0])))

    for split in range(len(splits)): 
        train_indices[split, :] = np.concatenate(splits[:split] + splits[split+1:])
        test_indices[split, :] = splits[split]
    return train_indices, test_indices




def boot_strap(*arrays, data_points, n_samples):
    # Bootstrap sepcified for use in 3d-regression, takes in two arrays  
    datasets = np.array(len(arrays))
    print(datasets)
    indices = np.random.randint(0, data_points, data_points)
    for i in range(len(arrays)): 
        datasets[i] = arrays[i][indices]
    return datasets
