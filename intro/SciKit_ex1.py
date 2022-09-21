
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02): 

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
    # marks the test examples 
    if test_idx: 
        # draws the chart off all examples 

        X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
        plt.scatter(X_test[:, 0], X[:, 1])

#------------------------------------------------------------------------------------------------------------------#


iris = datasets.load_iris() 
X = iris.data[:, [2, 3]] # Extracting trait nr. 2 and nr. 3 of every flower in the dataset 
y = iris.target
#print('Labels of classes:', np.unique(y))

""" To test the effectivness of a trained model for unknown data, we'll divide our dataset into separate 
sets, one for training and one for testing our model. We then use stratify=y to divide eaxh set into three groups 
of data, each of exatcly the same size."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# To show that we really have divided into sets with groups of equal size, we'll use bincount method from NumPy
print('Amount of labels in dataset y:', np.bincount(y))
print('Amount of labels in dataset y_train:', np.bincount(y_train))
print('Amount of labels in dataset y_test:', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())

# by importing a module from sklearn called metrics, the effectiveness of a machine learning model 
# can be measured in percent/decimal numbers.
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

# 