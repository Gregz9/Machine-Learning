from math import degrees
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from imageio.v2 import imread
from numpy.core import _asarray
from sklearn.metrics import mean_squared_error
from utils import ( 
    FrankeFunction, compute_optimal_parameters2, create_X, generate_determ_data, 
    compute_optimal_parameters, predict, R2, MSE, load_and_scale_terrain)
from Regression import Reg_model

def perform_ols(centering=False, degrees=20): 

    terrain_file = ('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_2.tif')
    terrain,N = load_and_scale_terrain(terrain_file)
    x, y = generate_determ_data(N)
    z = terrain[:N,:N]
    X = create_X(x,y,degrees, centering=centering)

    ols_reg = Reg_model(X=X, z=z, degree=degrees, reg_type='ols', centering=centering)

    for degree in range(1, degrees): 
        ols_reg.preprocess_data(degree=degree)
        ols_reg.fit_data()
        pred_train, pred_test = ols_reg.predict()
        MSE_train = ols_reg.cost_(ols_reg.z_train, pred_train)
        MSE_test = ols_reg.cost_(ols_reg.z_test, pred_test)

        print(MSE_train.shape)
        print(MSE_test.shape)
perform_ols()