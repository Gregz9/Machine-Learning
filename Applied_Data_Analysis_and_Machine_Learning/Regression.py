
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (generate_determ_data, create_X, FrankeFunction, MSE, compute_optimal_parameters,
                    compute_betas_ridge, predict) 
from sklearn import linear_model

class Reg_model:

    def __init__(self, X=None, z=None, lambas_=[], n_points=0, degree=0, reg_type='ols', resampling_type=None, scaling=False):
        self.reg_type = reg_type
        self.resampling_type = resampling_type
        self.scaling = scaling
        self.lambdas_ = lambas_
        self.n_points = n_points
        self.degree = degree
        
        if n_points == 0: 
            self.X = X 
            self.z = z
        else: 
            x,y =generate_determ_data(n_points)
            z = FrankeFunction(x, y, noise=True)
            X = create_X(x,y, self.degree)
            self.z = z
            self.X = X
        
    def cost_(self, test_data, pred_data): 
        return MSE(test_data, pred_data)


    def preprocess_data(self, degree=5, t_size=0.2): 
        """
        This method uses the scaling method called centering, which means that we remove first column of the 
        feature matrix in order to remove the intercept term. This is done when we cannot assume that the expected 
        outputs are zero when all predicators are zero. 
        """
        l = int((degree+1)*(degree+2)/2)
        
        if self.scaling: 
            self.X = self.X[: , 1:]
            self.x_train, self.x_test, self.z_train, self.z_test = train_test_split(self.X[:,:l], self.z.ravel(), test_size=t_size)

            self.intercept = np.mean(self.z_train)
            self.x_train -= np.mean(self.x_train)
            self.x_test -= np.mean(self.x_train)
        else: 
            self.x_train, self.x_test, self.z_train, self.z_test = train_test_split(self.X[:,:l], self.z.ravel(), test_size=t_size)


    def fit_data(self, X_train, y_train, lambda_=0): 
        if self.reg_type == 'ols': 
            self.betas = compute_optimal_parameters(X_train,y_train)
        elif self.reg_type == 'ridge': 
            self.betas = compute_betas_ridge(X_train,y_train,lambda_)
        elif self.reg_type == 'lasso': 
            self.RegLasso = linear_model.Lasso(alpha = lambda_, fit_intercept=False)
            self.RegLasso.fit(X_train, y_train)
    
    def predict(self): 
        if self.reg_type == 'ols' or self.reg_type == 'ridge': 
            if self.scaling: 
                return (predict(self.x_train, self.betas, self.intercept), 
                        predict(self.x_test, self.betas, self.intercept))
            else: 
                return predict(self.x_train, self.betas), predict(self.x_test, self.betas)
            
        else: 
            if self.scaling:  
                return (self.RegLasso.predict(self.x_train) + self.intercept, 
                        self.RegLasso.predict(self.x_test) + self.intercept)
            else: 
                return self.RegLasso.predict(self.x_train), self.RegLasso.predict(self.x_test)


    def bias_var_tradeOff(self, degree, pred_train, pred_test): 
        bias = np.mean(np.mean((self.z_train-pred_train)**2, axis=0, keepdims=True))
        variance = np.mean(np.mean((self.z_test-pred_test)**2, axis=0, keepdims=True))
        return bias, variance
    
            
if __name__ == '__main__':
    reg = Reg_model(n_points=10, degree=11, reg_type='ols')

    print(reg.X.shape) 
    reg.preprocess_data(degree=5)
    print(reg.X.shape)


    




    
        


