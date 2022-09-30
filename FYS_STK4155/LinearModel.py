
from utils import compute_optimal_parameters, MSE

class LinearModel(): 

    def __init__(self, reg_type, lambda_=0):

        self.reg_type = reg_type
        self.lamda_ = lambda_

    def fit(self, X, tar, intercept=False): 
        if len(tar.shape()) > 1: 
            tar = tar.ravel()

        self.X = X 
        self.targets = tar
        self.beta = compute_optimal_parameters(X, tar, intercept)
        return self

    def predict(self, X=None, beta=None):
        return X@beta

    def bootstrap(): 
        pass 


    



