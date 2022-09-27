import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler


fig = plt.figure()
fig2 = plt.figure() 
ax = fig.add_subplot(projection='3d')
ax2 = fig2.add_subplot(projection='3d')

# Making data ingestion to the function
x = np.arange(0, 1, 0.025)
y = np.arange(0, 1, 0.025) 
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
    return np.array(X) 

def compute_optimal_parameters(X, y): 
    # This method uses SVD to compote the optimal parameters beta for OSL.
    # SVD is chosen cause the central matrix of the expression for beta may 
    # cause problems if it's near-singular or singular. 
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    D = np.zeros((len(S), len(S)))
    for i in range(len(S)):
        D[i,i] = S[i]
    
    beta = (VT.T)@(np.linalg.inv(np.diag(S)))@(U.T)@(y.ravel())
    return beta


def optimal_parameters_inv(X, y): 
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T)
    #beta = beta.T

    beta = beta@(y.ravel())
    return beta


def perform_regression(X, beta): 

    franke_pred = np.array(())
    for i in range(X.shape[0]):

        franke_pred = np.append(franke_pred, np.sum(X[i]*beta))

    return franke_pred
    
def perform_manual_regression(x, y, beta): 
    pred = beta[0] + beta[1]*x + beta[2]*y + beta[3]*x**2 + beta[4]*y**2 + beta[5]*x*y
    return pred

X2 = generate_design_matrix(x, y, 5)
"""print('\n')
x2 = x.ravel() 
y2 = y.ravel()

X3 = np.stack((np.ones(len(x2)), x2, y2, x2**2, y2**2, x2*y2), axis=-1)
print(X3)

print(np.array_equal(X2, X3))
"""

z = FrankeFunction(x, y) 

beta_SVD = compute_optimal_parameters(X2, z)
beta_INV = optimal_parameters_inv(X2, z)

print(beta_INV)
print(beta_SVD)

preds_man = perform_manual_regression(x, y, beta_SVD)
preds = perform_regression(X2, beta_SVD)
preds = preds.reshape(len(x), len(y))

"""
x_and_y = np.hstack((x.ravel().reshape(x.ravel().shape[0],1), y.ravel().reshape(y.ravel().shape[0],1)))
scaler = StandardScaler()
scaler.fit(x_and_y)

X_scaled = scaler.transform(x_and_y)

X = generate_design_matrix(X_scaled.T[0], X_scaled.T[0], 2)
beta = compute_optimal_parameters(X, z)
"""

# Plot the surface of the function
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf2 = ax2.plot_surface(x, y, preds, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customization of z-axis
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax2.set_zlim(-0.10, 1.40)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors 
fig.colorbar(surf, shrink=0.5, aspect=5)
fig2.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
