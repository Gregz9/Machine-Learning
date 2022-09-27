import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure() 
ax = fig.add_subplot(projection='3d')

# Making data ingestion to the function
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05) 
x, y = np.meshgrid(x,y)

def FrankeFunction(x, y, noise=False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    stoch_noise = 0
    if noise: 
        stoch_noise = np.random.normal(0, 0.1, len(x)**2)
        stoch_noise = stoch_noise.reshape(len(x), len(x))

    return term1 + term2 + term3 + term4 + stoch_noise

def generate_design_matrix(x, y, order): 
    x = x.ravel() 
    y = y.ravel()
    
    X = np.array((np.ones(len(x))))
    for i in range(order): 
        if i == 0: 
            X = np.column_stack((X, x**(i+1), y**(i+1)))
        else: 
            X = np.column_stack((X, x**(i+1), y**(i+1), (x**i)*(y**i)))
    return X 

def compute_optimal_parameters(X, y): 
    # This method uses SVD to compote the optimal parameters beta for OSL.
    # SVD is chosen cause the central matrix of the expression for beta may 
    # cause problems if it's near-singular or singular. 
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    D = np.zeros((len(S), len(S)))
    for i in range(len(S)):
        D[i,i] = S[i]
    
    print('singular-value matrices')
    print(np.linalg.inv(D))
    print(np.linalg.inv(np.diag(S)))

    beta = (VT.T)@(np.linalg.inv(np.diag(S)))@(U.T)@(y.ravel())
    return beta


def optimal_parameters_inv(X, y): 
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T)
    #beta = beta.T

    beta = beta@(y.ravel())
    print(beta)


def perform_regression(X, beta): 

    franke_pred = np.array(())
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            
            franke_pred = np.append(franke_pred, np.sum(beta*X[i,:]))
    
    print(franke_pred.shape)


    


X2 = generate_design_matrix(x, y, 3)
print(X2.shape())
beta_SVD = compute_optimal_parameters(X2, y)
beta_INV = optimal_parameters_inv(X2, y)
perform_regression(X2, beta_SVD)

z = FrankeFunction(x, y) 


# Plot the surface of the function
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customization of z-axis
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors 
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
