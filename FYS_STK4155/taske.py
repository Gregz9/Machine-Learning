from ensurepip import bootstrap
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.pipeline import make_pipeline
from numpy.core import _asarray
from sklearn.utils import resample
import time
from utils import ( 
    FrankeFunction, generate_determ_data, create_X, create_simple_X,
    compute_optimal_parameters, generate_design_matrix, predict,
    compute_betas_ridge)



def Ridge_regression(n_points=20, degrees=5, n_boots=10, n_lambdas=10, scaling=False, noisy=True, r_seed=7): 
    np.random.seed(r_seed)
    x,y = generate_determ_data(n_points)
    z= FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees)
    lambdas = np.logspace(-4,4,n_lambdas)

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    bias = np.zeros(degrees)
    variance = np.zeros(degrees)
    polydegree = np.zeros(degrees)

    i, i2 = 3, 3
    for degree in range(1, degrees+1): 

        X_train, X_test, z_train, z_test = train_test_split(X[:, :i], z.ravel(), test_size=0.2)
        for lambdas_ in lambdas: 
            I = np.eye(i,i)
            pred_train_avg = np.empty((n_boots, z_train.shape[0]))
            pred_test_avg = np.empty((n_boots, z_test.shape[0]))

            for boot in range(n_boots):
            
            
            beta_ridge = compute_betas_ridge(X_train, z_train, lambdas[5]*I)
    

Ridge_regression()

