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

def generate_data(size): 
# Making data ingestion to the function
    x = np.arange(0, 1, 1/size)
    y = np.arange(0, 1, 1/size) 
    x, y = np.meshgrid(x,y)

    return x, y

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
    return np.array(X) 

def compute_optimal_parameters(A, y, intercept=True): 
    # This method uses SVD to compute the optimal parameters beta for OSL.
    # SVD is chosen cause the central matrix of the expression for beta may 
    # cause problems if it's near-singular or singular.
    U, S, VT = np.linalg.svd(A )#, full_matrices=False)
     
    d = np.divide(1.0, S, where=(S!=0))
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)

    beta = (VT.T)@(D.T)@(U.T)@y.ravel()
    return beta

def compute_optimal_parameters2(X, y):

    A = np.linalg.pinv(X)
    beta = A@y.ravel()

    return beta

def optimal_parameters_inv(X, y): 
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T)
    #beta = beta.T

    beta = beta@(y.ravel())
    return beta


def predict(X, beta, intercept=0): 

    franke_pred = np.array(())
    for i in range(X.shape[0]):
        franke_pred = np.append(franke_pred, np.sum(X[i]*beta) + intercept)
    
    return franke_pred
    
def perform_manual_regression(x, y, beta): 

    pred = beta[0] + beta[1]*x + beta[2]*y + beta[3]*x**2 + beta[4]*y**2 + beta[5]*x*y
    return pred

def R2(y, y_hat): 
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - np.mean(y_hat))**2)

def MSE(y, y_hat): 
    return np.sum((y-y_hat)**2)/np.size(y)

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

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def perform_OLS_regression(): 

    n_points = 40
    n = 5
    x, y = generate_data(n_points)
    
    for i in range(1, n): 
            
        X = generate_design_matrix(x, y, n)
        z = FrankeFunction(x, y) 

        x_train, x_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2, shuffle=True, random_state=42)

        scaler = MinMaxScaler()
        scaler.fit(x_train)
        
        #Centering datasets
        x_train_mean = np.average(x_train, axis=0) 
        y_train_mean = np.average(y_train, axis=0)     

        x_train_scaled2 = x_train - x_train_mean
        y_train_scaled2 = y_train - y_train_mean
        
        #Data scaled using SciKit-learn scaler
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        # Plot the surface of the function

        beta_SVD_scaled = compute_optimal_parameters(x_train_scaled2, y_train_scaled2)
        intercept = np.mean(y_train_mean - x_train_mean @ beta_SVD_scaled)
        
        preds2 = predict(X, beta_SVD_scaled)
        preds2 = preds2 + intercept
        preds2 = preds2.reshape(n_points, n_points)

        fig = plt.figure()
        axs = fig.add_subplot(1, 2, 1, projection='3d')
        surf = axs.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customization of z-axis
        axs.set_zlim(-0.10, 1.40)
        axs.zaxis.set_major_locator(LinearLocator(10))
        axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        axs.set_title("Frankes's function")

        axs = fig.add_subplot(1, 2, 2, projection='3d')
        surf = axs.plot_surface(x, y, preds2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        axs.set_zlim(-0.10, 1.40)
        axs.zaxis.set_major_locator(LinearLocator(10))
        axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        axs.set_title("Polynomial fit of n-th order to Franke's function")
        # Add a color bar which maps values to colors 
        fig.colorbar(surf, shrink=0.5, aspect=5)
        #fig2.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    
perform_OLS_regression()


