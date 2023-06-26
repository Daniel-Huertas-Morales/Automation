#Plotte Entscheidungsfunktion
def plot_decision_boundary(X,y,svm):
     
    fig = pl.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    #Scatter Plot
    ax.scatter(X[y==0,0],X[y==0,1],marker="x",zorder=100,color="k",label="Class 0")
    ax.scatter(X[y==1,0],X[y==1,1],marker="o",zorder=100,color="g", label="Class 1")
    
    #plot decision boundry
    pp = np.linspace(-3, 3, 1000)
    x0, x1 = np.meshgrid(pp, pp)
    X_ = np.c_[x0.ravel(), x1.ravel()]
    predictions = svm.predict(X_)
    decision_boundary = svm.decision_function(X_)
    
    ax.contourf(x0, x1, predictions.reshape(x0.shape), cmap="RdBu", alpha=0.3)
    ax.contourf(x0, x1, decision_boundary.reshape(x0.shape), alpha=0.2)
    
    #Set axis labels
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    #Set axis limits
    ax.set_ylim(-3,3)
    ax.set_xlim(-3,3)
    #show grid in grey and set top and right axis to invisible
    ax.grid(color="#CCCCCC")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    pl.legend()
    pl.tight_layout()