

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
from taskb import OLS_reg
from taskc import OLS_reg_boot
from taskd import OLS_reg_kFold
from taske import Ridge_reg_boot, Ridge_reg_kFold
from taskf import Lasso_reg_boot, Lasso_reg_kFold


def visualize_prediction(x,y,z,pred_vis, order):
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.terrain, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.75)
    ax.set_zlabel('Height [km]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(f"Real terrain")
    # -----------------------------------------------------------------------------------""
    ax = fig.add_subplot(1, 2, 2,projection='3d')
    ax.plot_surface(x, y, pred_vis, cmap=cm.terrain, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.75)
    ax.set_zlabel('Height [km]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
    ax.set_title(f"Polynomial fit of {order}-th order")
    plt.show()

if __name__ == '__main__':
    
    visualize_prediction(x,y,z,preds_cn[19], 20)

    terrain_file = ('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_2.tif')
    terrain,N = load_and_scale_terrain(terrain_file)
    x, y = generate_determ_data(N)
    z = terrain[:N,:N]
    OLS_reg(x,y,z=z)