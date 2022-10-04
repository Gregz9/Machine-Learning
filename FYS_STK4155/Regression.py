import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

class Regression:

    def __init__(self, X, z, lambas_,reg_type='ols', resampling_type=None, scaling=None):
        self.X = X
        self.z = z 
        self.reg_type = reg_type
        self.resampling_type = resampling_type
        self.scaling = scaling
        self.lambdas_ = lambas_

