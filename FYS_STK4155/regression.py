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

def generate_data(size, seed=None): 
# Making data ingestion to the function
    np.random.seed(seed)
    x = np.arange(0, 1, 1/size) #np.sort(np.random.uniform(0, 1, size))    
    y = np.arange(0, 1, 1/size) #np.sort(np.random.uniform(0, 1, size))
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

    if not intercept: 
        X = np.delete(X, 0, axis=1)
    return np.array(X) 

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
    A = np.linalg.pinv(X.T@X)@X.T
    beta = A@y.ravel()
    return beta

def compute_optimal_parameters_inv(X, y): 
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T)
    beta = beta@(y.ravel())
    return beta

def predict(X, beta, intercept=0): 
    franke_pred = np.array(())
    for i in range(X.shape[0]):
        franke_pred = np.append(franke_pred, np.sum(X[i]*beta) + intercept)

    return franke_pred

def R2(y, y_hat): 
    return 1 - np.sum((y - y_hat)**2) / np.sum((y - np.mean(y_hat))**2)

def MSE(y, y_hat): 
    n = np.size(y_hat)
    return np.sum((y-y_hat)**2)/n

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

def perform_OLS_regression(n_points=40, n=5, seed=None): 

    x, y = generate_data(n_points, seed)
    z = FrankeFunction(x,y)

    MSE_train_list = []
    MSE_test_list = []
    R2_train_list = []
    R2_test_list = []
    betas_list = []
    preds_cn = []

    for i in range(1, n+1): 
            
        X = generate_design_matrix(x, y, i, intercept=False)
        X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2, random_state=seed)
        
        #Centering datasets
        x_train_mean = np.mean(X_train, axis=0) 
        z_train_mean = np.mean(z_train, axis=0)     

        # Using centered values of X and y to compute parameters beta
        X_train_centered = X_train - x_train_mean
        z_train_centered = z_train - z_train_mean
        X_test_centered = X_test - x_train_mean 

        beta_SVD_cn = compute_optimal_parameters2(X_train_centered, z_train_centered)
        betas_list.append(beta_SVD_cn)

        intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)

        preds_visualization_cn = predict(X, beta_SVD_cn, intercept)
        preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
        preds_cn.append(preds_visualization_cn)

        preds_train_cn = predict(X_train_centered, beta_SVD_cn, z_train_mean) 
        preds_test_cn = predict(X_test_centered, beta_SVD_cn, z_train_mean)

        MSE_train_list.append(MSE(z_train, preds_train_cn))
        MSE_test_list.append(MSE(z_test, preds_test_cn))
        
        R2_train_list.append(R2(z_train, preds_train_cn))
        R2_test_list.append(R2(z_test, preds_test_cn))

    return betas_list, preds_cn, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list

def plot_figs(*args):
    
    fig, axs = plt.subplots(2,2)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']
    x = [i for i in range(1, len(args[0])+1)]
    y = np.zeros((len(x)))

    beta_matrix = np.zeros((len(args[0]), 5))
    for i in range(beta_matrix.shape[0]): 
        for j in range(len(args[0][i])):
            if j == 5:
                break
            beta_matrix[i][j] = args[0][i][j]
    for k in range(5): 
        axs[0,0].plot(x, [beta_matrix[i,k] for i in range(len(args[0]))], color_list[k], label=f'beta{k+1}')
    axs[0,0].plot(x, y, 'k--', label='x-axis')
    axs[0,0].set_xlabel('Polynomial order')
    axs[0,0].set_ylabel('Beta values')
    axs[0,0].legend()

    axs[0,1].plot(x, args[1], 'b', label='MSE_train') 
    axs[0,1].plot(x, args[3], 'r', label='MSE_test')
    axs[0,1].set_xlabel('Polynomial order')
    axs[0,1].set_ylabel('Mean Squared Error')
    axs[0,1].legend()

    axs[1,0].plot(x, args[2], 'g', label='R2_train')
    axs[1,0].plot(x, args[4], 'y', label='R2_test')
    axs[1,0].set_xlabel('Polynomial order')
    axs[1,0].set_ylabel('R2 Score')
    axs[1,0].legend()
    plt.show() 
    # ---------------------------------------------------------------------------------- #
    x, y = generate_data(args[6], seed)
    z = FrankeFunction(x,y)
    fig= plt.figure()
    ax = fig.add_subplot(1, 2, 1,projection='3d')
    surf = ax.plot_surface(x, y, args[5][4], cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customization of z-axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Frankes's function")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # -----------------------------------------------------------------------------------""
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Polynomial fit of n-th order")
    # Add a color bar which maps values to colors 
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
data_size = 20
betas, preds_cn, MSE_train, MSE_test, R2_train, R2_test = perform_OLS_regression(data_size ,n=10, seed=9)
print(preds_cn)
#print(MSE_train)
plot_figs(betas, MSE_train, R2_train, MSE_test, R2_test, preds_cn, data_size)

