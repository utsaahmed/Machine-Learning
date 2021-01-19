%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error
plt.rcParams.update({'font.size':22}) 
plt.rc('legend',fontsize=16)
plt.rcParams['figure.figsize']=(20,10)

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
scaler=StandardScaler()

nsample = 200
x = np.linspace(0, 10, nsample)
scaler.fit(x.reshape(-1,1))
#x=scaler.transform(x.reshape(1,-1))
x.reshape(1,-1)
# X = np.column_stack((x, x**2))
beta = np.array([50, 15])
e = 2*np.random.normal(size=nsample)
X = sm.add_constant(x.reshape(-1,1))
y = np.dot(X, beta) + e
# x
y.shape
x.reshape(-1,1).shape

x_train,x_test,y_train,y_test=train_test_split(x.reshape(-1,1),y,test_size=0.2,random_state=0)


trans=PolynomialFeatures(degree=2)
data=trans.fit_transform(x_train.reshape(-1,1))
data.shape


trans=PolynomialFeatures(degree=2)
data=trans.fit_transform(x_train.reshape(-1,1))

poly_reg=LinearRegression()
poly_reg.fit(data,y_train)
def viz_polymonial():
    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_test, poly_reg.predict(trans.fit_transform(x_test.reshape(-1,1))), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_polymonial()


poly_reg.coef_
# get_params.
a=poly_reg.get_params()


label_enc=preprocessing.LabelEncoder()
# x_train=scaler.transform(x_train.reshape(-1,1))
# y_train=scaler.transform(y_train.reshape(-1,1))
x_test=scaler.transform(x_test.reshape(-1,1))
y_test=scaler.transform(y_test.reshape(-1,1))
x_train=label_enc.fit_transform(x_train)
y_train=label_enc.fit_transform(y_train)
x_test=label_enc.fit_transform(x_test)
y_test=label_enc.fit_transform(y_test)

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train.reshape(-1,1), y_train)
y_pred = classifier.predict(x_test.reshape(-1,1))

def viz_polymonial():
    plt.scatter(x_test, y_test, color='red')
    plt.scatter(x_test, y_pred, color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_polymonial()
x_test.shape


from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
#loo.get_n_splits(x_train)
ans=[]
for train_index, test_index in loo.split(x_train):
  xt_train,xt_test=x_train[train_index],x_train[test_index]
  yt_train,yt_test=y_train[train_index],y_train[test_index]

  # print(xt_train,yt_train,xt_test,yt_test)

  classifier = KNeighborsClassifier(n_neighbors=5)
  classifier.fit(xt_train.reshape(-1,1), yt_train)
  yt_pred = classifier.predict(xt_test.reshape(-1,1))
  ans.append(mean_squared_error(yt_test,yt_pred))

print(np.average(ans))


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import plot_confusion_matrix
clf = DecisionTreeClassifier(random_state=0)

iris = load_iris()
x=iris.data[:,:3]
y=iris.target
#x=label_enc.fit_transform(x)
#y=label_enc.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

cross_val_score(clf, x_train,y_train, cv=10)
clf.fit(x_train,y_train)
#plot_tree(clf,filled=True)

tmp = np.linspace(-5,5,51)
x_plot,y_plot = np.meshgrid(tmp,tmp)

z=clf.predict(x_test)
z_proba=clf.predict_proba(np.hstack((y_plot,tmp)))

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, z(x_plot,y_plot))
ax.plot3D(x[y==0,0], x[y==0,1], x[y==0,2],'ob')
ax.plot3D(x[y==1,0], x[y==1,1], x[y==1,2],'sr')
ax.scatter3D(x[y==2,0], x[y==2,1], x[y==2,2],'green')
plt.show()
plot_confusion_matrix(clf,x,y)

X_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("utso") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names[:3],  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1
   xx1, xx2,xx3= np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution), np.arange(x3_min, x3_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel(),xx3.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contour3D(xx1, xx2,  Z, xx3,alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())
   plt.ylim(xx3.min(), xx3.max())

   # plot all samples
   X_test, y_test = X[test_idx, :], y[test_idx]
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=cmap(idx),
               marker=markers[idx], label=cl)
   # highlight test samples
   if test_idx:
      X_test, y_test = X[test_idx, :], y[test_idx]
      plt.scatter(X_test[:, 0], X_test[:, 1], c='',
               alpha=1.0, linewidth=1, marker='o',
               s=55, label='test set')
      

plot_decision_regions(X_combined, y_combined,
         classifier=clf, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import plot_confusion_matrix

# rs = np.random.RandomState(1234)

# # Generate some fake data.
# n_samples = 200
# # X is the input features by row.
# X = np.zeros((200,3))
# X[:n_samples/2] = rs.multivariate_normal( np.ones(3), np.eye(3), size=n_samples/2)
# X[n_samples/2:] = rs.multivariate_normal(-np.ones(3), np.eye(3), size=n_samples/2)
# # Y is the class labels for each row of X.
# Y = np.zeros(n_samples); Y[n_samples/2:] = 1

# Fit the data with an svm
x=iris.data[:,:3]
y=iris.target
#y=label_enc.fit_transform(y)

# x = x[np.logical_or(y==0,y==1)]
# y = y[np.logical_or(y==0,y==1)]

svc = SVC(kernel='linear')
svc.fit(x,y)

# The equation of the separating plane is given by all x in R^3 such that:
# np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
# to plot the plane in terms of x and y.

z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

tmp = np.linspace(-5,5,30)
x_plot,y_plot = np.meshgrid(tmp,tmp)

# Plot stuff.
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, z(x_plot,y_plot))
ax.plot3D(x[y==0,0], x[y==0,1], x[y==0,2],'ob')
ax.plot3D(x[y==1,0], x[y==1,1], x[y==1,2],'sr')
ax.scatter3D(x[y==2,0], x[y==2,1], x[y==2,2],'green')
plt.show()
plot_confusion_matrix(svc,x,y)

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
%matplotlib inline


iris = load_iris()
x=iris.data
y=iris.target

#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

sc=StandardScaler()
x=sc.fit_transform(x)

pca=PCA(n_components=3) 
principalComponents=pca.fit_transform(x) 
principalDf=pd.DataFrame(data=principalComponents,columns=['principal component 1','principal component 2','principal component 3']) 
#principalDf.head()
# pca2=incremen
plt.scatter(pca.explained_variance_,np.arange(1,4))

backtransform=pca.inverse_transform(principalComponents)

principalDfcorrect=pd.DataFrame(data=backtransform,columns=['principal component 1','principal component 2','principal component 3','principal component 4']) 
principalDfprev=pd.DataFrame(data=backtransform,columns=['principal component 1','principal component 2','principal component 3','principal component 4']) 


