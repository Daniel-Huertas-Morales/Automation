import matplotlib 
import pylab as pl
from sklearn.model_selection import cross_validate, GridSearchCV,StratifiedKFold
from sklearn.svm import SVC
import sklearn.metrics as metrics
import numpy as np


np.random.seed(0)

X=no.random.randn(200,2)
y=np.logical_xor(X[:,0]>0, X[:,1]>0)

fig=pl.figure(figsize=(10,5))
ax=fig.add_subplot(111)

ax.scatter(X[y==0,0],X[y==0,1], marker="x",zorder=100, color="k", alpha=0.7, label="Class 0")
ax.scatter(X[y==1,0],X[y==1,1], marker="x",zorder=100, color="g", alpha=0.7, label="Class 1")









Cs=np.logspace(-4,4,10)
print(Cs)

params={"C":Cs}
print(params)

kf=StratifiedKFold(n_splits=5)

best_test_acc=[]

for train_index, test_index in kf.split(X,y):
    X_train=X[train_index]
    X_test=X[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

    model=SVC(kernel="linear")
    gridsearch=GridSearchCV(model,params, cv=2, scoring="accuracy",iid=True) 
    gridsearch.fit(X_train,y_train) 

    y_pred=gridsearch.predict(X_test)
    test_acc=metrics.accuracy_score(y_test,y_pred)    
    best_test_acc.append(test_acc)

best_test_acc=np.array(best_test_acc)
print("Average ACC (Testing Data): %.2f (+- %.2f)" % (best_test_acc.mean(),best_test_acc.sd())) 


