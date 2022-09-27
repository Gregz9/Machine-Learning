import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn

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

    stoch_noise = 0
    if noise: 
        stoch_noise = np.random.normal(0, 0.1, len(x)**2)
        #stoch_noise = stoch_noise.reshape(len(x), len(x))

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
 
    #beta = U@S@VT@y.ravel()
    beta = (VT.T)@(np.linalg.inv(np.diag(S)))@(U.T)@(y.ravel())
    #beta = (VT.T)@(D)@(U.T)
    return beta


def optimal_parameters_inv(X, y): 
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T)
    #beta = beta.T

    beta = beta@(y.ravel())
    return beta


def predict(X, beta): 

    franke_pred = np.array(())
    for i in range(X.shape[0]):

        franke_pred = np.append(franke_pred, np.sum(X[i]*beta))

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
    x, y = generate_data(n_points)
 
    X2 = generate_design_matrix(x, y, 2)
    z = FrankeFunction(x, y) 
    print(z.shape)
  
    x_train, x_test, y_train, y_test = train_test_split(X2, z.ravel(), test_size=0.2, shuffle=False)

    beta_SVD = compute_optimal_parameters(X2, z)
    beta_INV = optimal_parameters_inv(X2, z)

    preds = predict(X2, beta_SVD)
    preds = preds.reshape(len(x), len(y))

    beta_SVD_scaled = compute_optimal_parameters(x_train, y_train)
    preds2 = predict(x_train, beta_SVD_scaled)
    #preds2 = preds2.reshape(n_points, n_points)

    x_and_y=np.hstack((x.ravel().reshape(x.ravel().shape[0],1),y.ravel().reshape(y.ravel().shape[0],1)))
    print(x_and_y.shape)

    
    """
    x_and_y = np.hstack((x.ravel().reshape(x.ravel().shape[0],1), y.ravel().reshape(y.ravel().shape[0],1)))
    scaler = StandardScaler()
    scaler.fit(x_and_y)

    X_scaled = scaler.transform(x_and_y)

    X = generate_design_matrix(X_scaled.T[0], X_scaled.T[0], 2)
    beta = compute_optimal_parameters(X, z)
    """

    # Plot the surface of the function
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

def test():
    # Making meshgrid of datapoints and compute Franke's function
    n = 5
    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    z = FrankeFunction(x, y)
    X = create_X(x, y, n=n)    
    # split in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.2)
    print(X.shape)

perform_OLS_regression()

#test()