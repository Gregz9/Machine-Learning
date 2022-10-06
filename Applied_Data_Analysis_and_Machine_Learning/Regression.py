
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (generate_determ_data, create_X, FrankeFunction, MSE, compute_optimal_parameters,
                    compute_betas_ridge, predict) 
from sklearn import linear_model

class Reg_model:

    def __init__(self, X=None, z=None, lambas_=[], n_points=0, degree=0, reg_type='ols', resampling_type=None, scaling=None):
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

    def fit_data(self, X_train, y_train, lambda_=0): 
        if self.reg_type == 'ols': 
            self.betas = compute_optimal_parameters(X_train,y_train)
        elif self.reg_type == 'ridge': 
            self.betas = compute_betas_ridge(X_train,y_train,lambda_)
        elif self.reg_type == 'lasso': 
            self.RegLasso = linear_model.Lasso(alpha = lambda_, fit_intercept=False)
            self.RegLasso.fit(X_train, y_train)
    
    def predict(self, X_test, betas=0): 
        if self.reg_type == 'ols' or self.reg_type == 'ridge': 
            return predict(X_test, betas)
        else: 
            return self.RegLasso.predict(X_test)
    
            

    




    
        


