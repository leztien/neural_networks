
"""
MPL Classifier with gradient checking and other things

"""


import numpy as np
np.set_printoptions(suppress=True)


def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module


#===========================================================


from sklearn.base import BaseEstimator, ClassifierMixin

class MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, activation=None, lagrangian=None, learning_rate=None,
                 batch_size=None, validation=None, max_iter=None, verbose=None, seed=None):
        #validate the hyperparameters
        if not any(isinstance(layers,cls) for cls in (tuple,list))\
            or len(layers)<1 or not all(isinstance(e,int)and e>2 for e in layers):
            raise TypeError("layers must be tuple or list of positive integers")
        
        if (lagrangian is not None) and lagrangian < 0: raise ValueError("lagrangian must be greater than zero") 
        if not(validation in (None,False) or 0<validation<0.5): raise ValueError("bad validation proportion")
        
        
        self.layers = tuple(layers)
        self.activation = get_activation_function(activation)
        self.output_activation = None
        self.lagrangian = 0.0 if lagrangian in (None,False) else np.float64(1e-9)    # for no penalty set lagrangian=0
        self.learning_rate = np.float64(learning_rate or 0.001)
        self.batch_size = abs(int(batch_size or 0))
        self.validation = validation
        self.max_iter = abs(int(max_iter or 300))
        self.verbose = verbose
        self.seed = seed
        
      
 
        
    def fit(self, X,y):
        #shapes, classes
        m,n = X.shape
        classes = sorted(set(y))
        K = len(classes)
        shapes = (n,) + self.layers + ((1 if K==2 else K),)
        shapes = [(m,n) for m,n in zip(shapes[1:],shapes[:-1])]
        L = len(shapes)
        
        #reshape the target
        if K==2:
            Y = np.array(y, dtype="uint8").reshape(1, len(y))
        else:
            Y = np.zeros(shape=(K,len(y)), dtype="uint8")
            [Y.__setitem__((y,i),1) for i,y in zip(range(len(y)),y)]
        Yfull = Y   # for batch_size in SGD

        
        #if validation:
        if self.validation:
            nx = np.random.permutation(m)
            Y = Y.T
            X,Y = (nd[nx] for nd in (X,Y)) #shuffling
            ix = int(len(y) * (1-self.validation)) # index where the validation subset should start
            (X,Y),(Xval,Yval) = (nd[:ix] for nd in (X,Y)), (nd[ix:] for nd in (X,Y))
            Y = Y.T
            Yval = Yval.T
            Yfull = Y
            m = len(X)
            
        
        #determine the output activation function (based on the output shape)
        self.output_activation = get_activation_function("sigmoid" if K==2 else "softmax")
        
        #initializers (i.e. standard deviations of the random weight-matreces per layer):
        initializers = [self.activation.initializer(n_out,n_in) for (n_out,n_in) in shapes[:-1]]
        initializers += [self.output_activation.initializer(*shapes[-1]),]  # add the output layer
        
        #create random weight-matreces and biases
        if self.seed: 
            if self.seed is True:
                from random import randint
                self.seed = randint(0, int(2**30))
                print("random seed for NN weights =", self.seed)
            np.random.seed(abs(int(self.seed)))
        W = [0,]+[np.random.randn(m,n)*σ for (m,n),σ in zip(shapes,initializers)]
        b = [0,]+[np.zeros(shape=(m,1), dtype="float64") for (m,n) in shapes]
        
        #create containers (for A,Z,G,g)
        A = [X.T,]+[None,]*L    
        Z = [0,] + [None,]*L
        G = [0,] + [None,]*L
        g = [0,] + [None,]*L
        
        #create logs for cost
        self.train_cost = list()
        self.validation_cost = list()
        
        #LOOP
        for epoch in range(self.max_iter):
            #if SGD
            if self.batch_size:
                nx = np.random.permutation(m)[:self.batch_size]
                A[0] = X[nx].T
                Y = Yfull[:,nx]

            #forward propagation
            for l in range(1, L+1):
                Z[l] = np.matmul(W[l], A[l-1]) + b[l]
                A[l] = (self.activation if l<L else self.output_activation)(Z[l])
            
            #back propagation
            for l in range(L,0,-1):
                Δ = (A[L]-Y) if l==L else np.matmul(W[l+1].T, Δ) * self.activation(Z[l], derivative=True)
                G[l] = ( np.matmul(Δ, A[l-1].T)  +  self.lagrangian * W[l]) / m
                g[l] = Δ.sum(axis=1, keepdims=True) / m
            

            #gradient check
            gradient_check = self.verbose >= 3
            if gradient_check and not(epoch%100) and epoch>1 and not self.batch_size and not self.validation:
                t = self.gradient_checking(W,b, G,g, X,Y)
                print("\ngradient check: {:.0%} of weights match. The average vectors distance is {:.15f}".format(*t))
                          
            #update weights
            for l in range(1, L+1):
                W[l] -= self.learning_rate * G[l]
                b[l] -= self.learning_rate * g[l]
                self.weights = W[1:]
                self.bias = b[1:]
                           
            #calculate and log cost
            J = self._cost(Y, A[L])
            self.train_cost.append(J)

            if self.validation:
                Pval = self.probabilities(Xval, weights=W, bias=b)
                Jval = self._cost(Yval, Pval.T)
                self.validation_cost.append(Jval)
            
            #check convergence
            if self.validation:
                n_last = 10
                if len(self.validation_cost) > n_last:
                    l = self.validation_cost[-n_last:]
                    no_change = np.allclose(l, np.mean(l), rtol=0.001)
                    if no_change and Jval<0.09:
                        print(f"The validation cost hasn't changed significantly for {n_last} last epochs. Breaking out after epoch#{epoch}")
                        break
            else:
                if J < 0.02:
                    print("The cost is {:.3f}. Breaking out after epoch#{}".format(J,epoch))
                    break
            
            #printing current report
            if not(epoch%100) and self.verbose:
                if K==2:
                    acc_train = np.equal((A[L]>.5).astype("uint8").ravel(), Y.ravel()).mean()
                    if self.validation: 
                        P = self.predict(Xval, weights=W, bias=b)
                        acc_val = np.equal(P, Yval.ravel()).mean()
                else:
                    acc_train = np.equal(A[L].argmax(axis=0), Y.argmax(axis=0)).mean()
                    if self.validation: 
                        P = self.predict(Xval, weights=W, bias=b)
                        acc_val = np.equal(P, Yval.argmax(axis=0)).mean()

                if self.validation:
                    msg = "train/val cost: {:.3f}/{:.3f}\ttrain/val accuracy: {:.1%}/{:.1%}\tepoch: {}".format(J,Jval, acc_train, acc_val, epoch)
                else:
                    msg = "train cost = {:.3f}\ttrain accuracy = {:.1%} \tepoch {}".format(J, acc_train, epoch)
                print(msg)
        
        #if not converged:
        else:
            from warnings import warn
            msg = "Increase max_iter. {} epochs is not enough.".format(str(self.max_iter))
            warn(msg, Warning)

        #after the loop
        self.weights = W[1:]
        self.bias = b[1:]
        return self
        #---end of fit method-------------------------------------------------
        

    def _cost(self, Y, P):  # Y = Ypred, P = Ppred
        if P.ndim==1: P = P.reshape(1,len(P))
        m = max(*P.shape)
        
        if Y.shape != P.shape: P = P.T
        assert Y.shape==P.shape,"inconsistent shapes: {} {}".format(Y.shape, P.shape)
        
        penalty = 0 if not self.lagrangian else (sum(W.sum() for W in self.weights) * (self.lagrangian/m))
        
        if P.shape[0]==1 or P.shape[1]==1:
            y, p = Y.ravel(), P.flatten()
            J = -(np.dot(np.log(p), y) + np.dot(np.log(1-p), (1-y))) / m + penalty
        else: J = -(np.log(P)*Y).sum() / m  + penalty
        return J
    
    
    def probabilities(self, X, **kwargs):
        
        W = kwargs.get("weights", None) or self.weights
        b = kwargs.get("bias", None) or self.bias
        if not isinstance(W[0], np.ndarray): W = W[1:]
        if not isinstance(b[0], np.ndarray): b = b[1:]
        
        L = len(W)
        A = X.T
        
        for l in range(L):  #note: here - zero based
            Z = np.matmul(W[l], A) + b[l]
            A = (self.activation if l<(L-1) else self.output_activation)(Z)
        P = A.ravel() if A.shape[0]==1 else A.T
        return(P)
    
    
    def predict(self, X, **kwargs):
        P = self.probabilities(X, **kwargs)
        if P.ndim==1:
            ypred = (P > 0.5).astype("uint8")
        elif P.ndim==2:
            ypred = P.argmax(axis=1)
        else: raise TypeError("bad dims")
        return ypred
    
    
    def gradient_checking(self, W,b, G,g, X,Y):  #checks weights only (not biases)
        from numpy.linalg import norm           
        L = len(W[1:])
        ε = 0.0001
        run = ε*2
        tol = 0.001
        counter = list()
        GG = [np.zeros_like(G) for G in G]  # GG = Gradients matreces of manual partials
        
        for l in range(1, L+1):
            for i in range(W[l].shape[0]):
                for j in range(W[l].shape[1]):
                    W0 = [l.copy() if isinstance(l, np.ndarray) else 0 for l in W]
                    W1 = [l.copy() if isinstance(l, np.ndarray) else 0 for l in W]
                    W0[l][i,j] -= ε
                    W1[l][i,j] += ε
                    P0 = self.probabilities(X, weights=W0, bias=b)
                    P1 = self.probabilities(X, weights=W1, bias=b)
                    J0 = self._cost(Y,P0)
                    J1 = self._cost(Y,P1)
                    rise = J1-J0
                    partial = rise/run
                    GG[l][i,j] = partial
                    counter.append(np.allclose(partial, G[l][i,j], rtol=tol))

        p = np.mean(counter)
        #use the vectors distance formula:
        vector_distances = list()
        for l in range(1, L+1):
            for i in range(G[l].shape[0]):
                d = norm(G[l][i] - GG[l][i])/norm(G[l][i] + GG[l][i])
                vector_distances.append(d)
        d = sum(vector_distances)/len(vector_distances)
        return(p,d)



#===================================================================

def get_activation_function(name):
    d = dict()
        
    def register(func):
        nonlocal d
        name = func.__name__.lower().strip()
        d.setdefault(name, func)
        func.initializer = lambda n_out,n_in: np.sqrt( (1/n_in)if(name=="tanh") else(2/n_in)if("relu"in name) else (2/(n_in*n_out)))
        return func
    
    #paste your activation functions here:
    @register
    def linear(Z, derivative=False):
        if derivative: return np.ones_like(Z, dtype=float)
        return Z

    @register
    def relu(Z, derivative=False):
        if derivative: return (Z>0).astype(float)
        return np.maximum(Z,0)
    
    @register
    def leakyrelu(Z, derivative=False):
        leak = 0.01
        if derivative: return np.where(Z>0, 1, leak)
        return np.maximum(Z, Z*leak)
    
    @register
    def tanh(Z, derivative=False):
        if derivative: return 1 - np.tanh(Z)**2
        return np.tanh(Z)
    
    @register
    def sigmoid(Z, derivative=False):
        if derivative: return np.exp(-Z) / (1 + np.exp(-Z))**2
        return 1 / (1 + np.exp(-Z))
    
    @register
    def softmax(Z, derivative=False):
        if derivative: return None
        return np.exp(Z) / np.exp(Z).sum(axis=0, keepdims=True)
    
    #provide for double names (of activation functions)
    t = (
    ['logistic','sigmoid'],
    ['leaky','leakyrelu','leaky-relu','leaky_relu'],
    ['tanh','tangent hyporbolic'],
    ['linear','identity']
    )
    
    for i in range(len(t)):
        l = [s for s in d.keys() if s in t[i]]
        if l:
            t[i].remove(l[0])
            [d.__setitem__(k, d[l[0]]) for k in t[i]]
    
    #get the appropriate activation function from the dictionary
    if name is None:
        return d["sigmoid"]
    else:
        try: return d[str(name).lower().strip()]
        except KeyError:
            msg = "No such activation function: " + str(name)
            raise ValueError(msg)



####################################################################################

from sklearn.model_selection import train_test_split

url = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_ANN.py"
#module = load_from_github(url)
X,y = module.make_data_for_ANN(m=1000, n=5, K=3, L=2, u=16,
                               balanced_classes=True, space_between_classes=True, 
                               gmm=False, seed=None)


md = MLP(layers=(64,32,16), activation="tanh", lagrangian=0.1, learning_rate=0.1,
         batch_size=128, validation=0.2, max_iter=9000, verbose=2, seed=True)


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)


md.fit(Xtrain, ytrain)

P = md.probabilities(Xtrain)
ypred = md.predict(Xtest)

acc = md.score(Xtest, ytest)
print("test accuracy =",acc)

import matplotlib.pyplot as plt
plt.plot(md.train_cost)
if md.validation_cost: plt.plot(md.validation_cost)



