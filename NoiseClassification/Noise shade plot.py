# Space for importing required libraries
from sklearn.datasets import make_classification, make_circles #libraries to generate data
import matplotlib.pyplot as plt # importing library to plot
from sklearn.linear_model import LogisticRegression #importing Logistic Regression 
import numpy as np #numpy library for data manipulation and plotting
from  sklearn.neural_network import MLPClassifier #to use neural net
import pandas as pd
# =============================================================================
#  np.random.seed(4)
# =============================================================================
data = pd.read_csv('Noisedatanight.csv')
y=[]
X=[]
a = (data.iloc[:,0:3].values)
for i in a:
    X.append([i[0],i[1]])
    if(i[2]=='L'):
        y.append(2)
    elif(i[2]=='M'):
        y.append(1)
    else:
        y.append(0)
X=np.array(X)    
y=np.array(y, dtype="int64")
 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = (scaler.transform(X))

f, ax = plt.subplots(figsize=(8, 8)) #plotting data
ax.scatter(X[:,1],X[:,0],c=y , s=50,cmap="RdBu")
ax.get_figure()


# =============================================================================
# architecture = (64,64,32,2) ###(A num_layers sized tuple with number of hidden neurons as each element)
# activationf = 'relu'
# learning_rate=0.01
# mlp = MLPClassifier(hidden_layer_sizes=architecture,activation=activationf,learning_rate_init=learning_rate)
# 
# =============================================================================

# =============================================================================
# from sklearn import linear_model
# clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
# 
# =============================================================================

from sklearn import svm
clf = svm.SVC(gamma='scale',max_iter=-1,C=1000)


clf.fit(X,y) #training classifier

# Predictions of trained classifiers to plot the decision plane
xx=np.linspace(min(X[:,1]), max(X[:,1]),1000) #bounds of data
yy=np.linspace(min(X[:,0]), max(X[:,0]),1000)


#Colour coding
X11=[]
X22=[]
pred=[]
for point1 in xx:
    for point2 in yy:
        X11.append(point1)
        X22.append(point2)
        pred.append(clf.predict(np.array([point1,point2]).reshape(1,-1))) #getting predictions across each point
pred = [int(i) for i in pred]        



#Plotting
f, ax1 = plt.subplots(figsize=(8, 8))
#ax1.contour(xx, yy, pred, levels=[0.5], cmap="Greys", vmin=0, vmax=.6)
ax1.scatter(X11, X22, s=50,c=pred, linewidth=1,cmap="RdYlGn")

# =============================================================================
# ax1.scatter(X[:,1], X[:, 0], c=y, s=50,
#              cmap="RdGn", vmin=-.2, vmax=1.2,
#              edgecolor="white", linewidth=1)
#  
# =============================================================================
ax1.set(aspect="equal",
       xlim=(min(X[:,1]), max(X[:,1])), 
       ylim=(min(X[:,0]), max(X[:,0])),
       xlabel="$X_1$", ylabel="$X_2$")


ax1.get_figure()

plt.savefig('night.png')
