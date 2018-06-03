# Check versions of libraries

# # Python version
# import sys
# print("Python: {}".format(sys.version))
# # scipy
# import scipy
# print("Scipy: {}".format(scipy.__version__))
# # numpy
# import numpy
# print("Numpy: {}".format(numpy.__version__))
# # matplotlib
# import matplotlib
# print("Matplotlib: {}".format(matplotlib.__version__))
# # pandas
# import pandas
# print("Pandas: {}".format(pandas.__version__))
# # scikit-learn
# import sklearn
# print("Sklearn: {}".format(sklearn.__version__))

# load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url,names=names)

# # shape
# print(dataset.shape)
# print("")
# # head
# print(dataset.head(20))
# print("")
# # descriptions
# print(dataset.describe())
# print("")
# # class distributions
# print(dataset.groupby('class').size())
# print("")

# # box and whisker plots
# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# # histograms
# dataset.hist()
# # scatter plot martrix
# scatter_matrix(dataset)

# plt.show()

# split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
