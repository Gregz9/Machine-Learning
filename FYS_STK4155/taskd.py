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
    FrankeFunction, generate_determ_data, create_X, compute_optimal_parameters, 
    compute_optimal_parameters_inv, generate_design_matrix, predict, MSE, KFold_split,
)

def OLS_cross_reg(n_points=20, degrees=5, scaling=False, noisy=True): 
    
    x,y = generate_determ_data(n_points)
    z = FrankeFunction(x,y,noise=noisy)
    X = create_X(x,y,degrees)
    z = z.ravel()
    train_indices, test_indices = KFold_split(z,4)
    print(np.int64(train_indices[0]))


    x_train, z_train = X[np.int64(train_indices[0]), :], z[np.int64(train_indices[0])] 


OLS_cross_reg()