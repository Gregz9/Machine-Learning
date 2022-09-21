
import numpy as np 
import os 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 

def plot_decision_regions(X, y, classifier, resolution=0.02): 

    # Configuring generator of markers and kolors 
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Drawing chart of decision area
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    

    # Draws the chart of examples 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y ==cl, 0], y=X[y == cl, 1], 
                alpha=0.8, color=colors[idx], 
                marker=markers[idx], label = cl,
                edgecolor='black')

class AdalineGD(object): 
    """Classificator - ADAptive LInear NEuron

    Parameters 
    -----------
    eta: floating point variable 
        Learning factor (in the intervall of 0.0 to 1.0).
    n_iter: integer variable 
        Number of iterations thorugh dataset used for training the algorithm. 
    random_state: integer variable 
        Seed of the generator of random numbers used to generate random wieghts 
        (We use a seed to be able to reuse the same set of pseudo-random/accidental numbers). 
    
    Attributes
    ----------
    w_ : table of one dimension (Table can be understood as column-vector)
        Weights to be fit 
    cost_ : list 
        Sum of squared errors (value og the cost function) for each era/iteration
    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1): 
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state 

    def fit(self, X, y): 
        """ Method used for teaching the algorithm using training data

        Parameters
        ----------
        X : {table-alike}, dimensions = [n_exaples, n_cech] (Matrix/vector)
            Learning vector 
        y : table-alike, dimensions = [n_examples]
            target values (final correct answers when testing)

        Output
        ------
        self:object 
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter): 
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 
            self.cost_.append(cost)
        return self 

    def net_input(self, X): 
        """Calculates the total stimulation"""
        return X

    def activation(self, X): 
        """Calculates the linear function of activation"""
        return np.dot(X, self.w_[1:]) + self.w_[0]          

    def predict(self, X): 
        """ Output: Returns the label of a class after peroforming a unit jump""" 
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

if __name__ == "__main__": 

    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    print('URL Address: ', s)
    df = pd.read_csv(s, header=None, encoding='utf-8')
    print(df.tail())

    # We are choosing types of iris flowers 'setosa' and 'versicolor'
    y = df.iloc[0:100, 4].values 
    y = np.where(y == 'Iris-setosa', -1, 1)

    # length of sepal and flower flake length 
    X = df.iloc[0:100, [0, 2]].values

    # example of how a badly chosen learning factor can affect the cost function of an classification algorithm 
    """fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = AdalineGD(n_iter = 10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) +1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Eras')
    ax[0].set_ylabel('Log (sum of squared errors')
    ax[0].set_title('Adaline - Learning factor of 0.01')
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) +1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Eras')
    ax[1].set_ylabel('Sum of squared errors')
    ax[1].set_title('Adaline - Learning factor of 0.0001')
   # plt.show()"""

   # To optimize the method of gradient decent, we standarize the values of X 
   # We do that by simply using methods built into NumPy library 

    """X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    ada_gd = AdalineGD(n_iter=15, eta=0.01)
    ada_gd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - gradient decent method')
    plt.xlabel('Length of flower sepal [standarized]') # Standarized as in the statisical understanding 
    plt.ylabel('Length of flower petal [standarized]')
    plt.legend(loc='upper left')
    plt.show() 

    plt.plot(range(1, len(ada_gd.cost_) +1), ada_gd.cost_, marker='o')
    plt.xlabel('Eras')
    plt.ylabel('Sum of squared errors')
    plt.tight_layout()
    plt.show()"""

    # Next example of a possible improvement for the method of gradient decent 
    # is to use it's stochastic verison
    



