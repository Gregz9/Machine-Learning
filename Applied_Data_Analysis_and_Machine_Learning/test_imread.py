
import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
from utils import create_X, load_and_scale_terrain
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def visualize_prediction(x,y,z):
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.terrain, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.75)
    ax.set_zlabel('Height [km]')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    #fig.colorbar(surf, shrink=0.5
    


# Load the terrain
terrain = imread('C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_1.tif')
terrain_file = 'C:\\Users\gregor.kajda\OneDrive - insidemedia.net\Desktop\Project_1\Machine-Learning\Applied_Data_Analysis_and_Machine_Learning\Data\SRTM_data_Norway_1.tif'
terrain, n = load_and_scale_terrain(terrain_file)


m = 10 # polynomial order


# Creates mesh of image pixels
x = np.linspace(0,1, n)
y = np.linspace(0,1, n)
x_mesh, y_mesh = np.meshgrid(x,y)

#z = terrain/1000
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#X = create_X(x, y, z)
visualize_prediction(x_mesh,y_mesh,terrain)


