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
from imageio.v2 import imread 
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

def create_X(x, y, n, centering=False):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    N = len(x)
    l = int((n+1)*(n+2)/2)
    X = np.zeros((N,l))
    
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    if centering: 
        return X[:,1:]
    else:         
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

def compute_betas_ridge(X, y, lambda_):
    lambda_ = lambda_*np.eye((X.T@X).shape[0], (X.T@X).shape[1])
    A = np.linalg.pinv(X.T@X + lambda_)@X.T
    beta = A@y.ravel()
    return beta

def compute_lasso_parameteres(X,y, lambda_): 
    pass

def lasso_penalty(self, beta, lambda_): 
    coef_l = lambda_*np.sum(np.abs(beta))
    return coef_l*np.sign(beta)

def predict(X, beta, intercept=0): 
    franke_pred = np.array(())
    for i in range(X.shape[0]):
        franke_pred = np.append(franke_pred, (X[i]@beta) + intercept)

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
    split_indices = np.zeros(k-1, dtype=np.int64)
    split_size = int(np.ceil(len(z)/k))
    length_ = split_size*(k-1)

    split_indices[:k-2] = np.arange(length_/(k-1), length_, length_/(k-1), dtype=np.int64)
    split_indices[k-2] = length_

    indices = np.arange(0,len(z),1)
    splits = np.split(indices, indices_or_sections=split_indices)

    train_indices = [] 
    test_indices = [] 

    for split in range(len(splits)): 
        size = 0 
        a = []
        b = []
        for i in range(len(splits)):
            if i == split: 
                b.extend(splits[i])
                continue
            size += len(splits[i])
            a.extend(splits[i][:])
        
        train_ind = np.array(a, dtype=np.int64)
        test_ind = np.array(b, dtype=np.int64)
    
        train_indices.append(train_ind)
        test_indices.append(test_ind)
        
    return (train_indices, test_indices)

def load_and_scale_terrain(filename):

    terrain = imread(filename)

    if terrain.shape[0] > terrain.shape[1]:
        terrain = terrain[:terrain.shape[1], :]
    elif terrain.shape[0] < terrain.shape[1]:
        terrain = terrain[:, :terrain.shape[0]]
        
    quarter = terrain.shape[0]//4
    terrain = terrain[quarter: quarter+300, quarter:quarter+300]
    #terrain = terrain[0:-1:slice, 0:-1:slice]
  
    print(filename, 'loaded.', terrain.shape[0],'x',terrain.shape[1])
    return terrain/1000, terrain.shape[0]


def boot_strap(data_points,*arrays):
    # Bootstrap sepcified for use in 3d-regression, takes in two arrays  
    datasets = np.array(len(arrays))
    indices = np.random.choice([i for i in range(arrays[0].shape[0])], arrays[0].shape[0], replace=True)

    for array in arrays: 
        array = array[indices]
        print(array)
    return arrays

