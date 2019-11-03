"""
demo: MPL classifier (my first NN, not abstracted into a class)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_data(layers, activation, seed=None):
    #activation functions
    identity = lambda Z : Z
    relu = lambda Z : np.maximum(0, Z)
    sigmoid = lambda Z : 1 / (1 + np.exp(-Z))
    tanh = lambda Z : np.tanh(Z)
    activation = locals().get(str(activation), None)
    
    #shapes
    m = 2000
    n = layers[0]
    shapes = [(b,a) for a,b in zip(layers[::],layers[1::])]
    
    np.random.seed(seed or np.random.randint(0, 2**30))
    WW = [np.random.uniform(-10,10, size=shape) for shape in shapes]
    bb = [np.random.uniform(-10,10, size=(shape[0],1)) for shape in shapes]

    r = radius = 1
    r = np.linspace(-r,r, max(3, round(m**(1/n))))
    X = np.array([nd.ravel() for nd in np.meshgrid(*[r]*n)]).T
    mask = (X**2).sum(1)**0.5 <= radius * (n-1)
    X = X[mask]
    
    #scale, transpose
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    A = X.T  # A[0]
    
    #forward propagation
    for l in range(len(bb)):
        W,b = WW[l], bb[l]
        Z = np.matmul(W,A) + b
        A = (sigmoid if (l==len(layers)-1) else activation)(Z)
    
    p = A.ravel()
    threshold = np.quantile(p, q=0.5)   # for balanced dataset
    y = (p >= threshold).astype("uint8")
    return(X,y)


#activation functions 
def sigmoid(Z, derivative=False, input_is_activation=False, **kwargs):
    if not derivative: return 1 / (1 + np.exp(-Z))
    elif derivative and input_is_activation:
        A = Z
        return A*(1-A)
    else: return np.exp(-Z) / (1+np.exp(-Z))**2

def tanh(Z, derivative=False):
    if not derivative: return np.tanh(Z)
    else: return 1 - np.tanh(Z)**2

def relu(Z, derivative=False):
    if not derivative: return np.maximum(0, Z)
    else: return (Z > 0).astype("float64")
    
    

def forward(W,b, Z, A, activation):
    sigmoid = lambda Z : 1 / (1 + np.exp(-Z))
    L = len(W)-1
    for l in range(1, L+1):
        Z[l] = np.matmul(W[l], A[l-1]) + b[l]
        A[l] = (sigmoid if l==L else activation)(Z[l])
    return A[l]



def gradient_check(W,b, G,g):
    ε = 0.01
    run = ε*2
    counter = 0
    
    for frame in range(1, L+1):
        R,C = [nd.ravel() for nd in np.indices(W[frame].shape)]
        indeces = tuple(zip(R,C))
        #checking the Weights matreces:
        for nx in indeces:
            W0 = np.array([W.copy() for W in W], copy=True)
            W1 = np.array([W.copy() for W in W], copy=True)
    
            W0[frame][nx] -= ε
            W1[frame][nx] += ε
            
            P0 = forward(W0, b, Z, A, activation=activation)
            P1 = forward(W1, b, Z, A, activation=activation)
            J0 = -(np.log(P0)*Y + np.log(1-P0)*(1-Y)).mean()
            J1 = -(np.log(P1)*Y + np.log(1-P1)*(1-Y)).mean()
            rise = J1-J0
            manual_derivative = rise/run
            backprop_derivative = G[frame][nx]
            b_ = np.allclose(manual_derivative,backprop_derivative, rtol=7e-1)
            counter += int(not b_)    
        #checking the bias vectors:
        for ix in range(len(b[frame])):
            b0 = np.array([b.copy() for b in b], copy=True)
            b1 = np.array([b.copy() for b in b], copy=True)
            
            b0[frame][ix] -= ε
            b1[frame][ix] += ε
            
            P0 = forward(W, b0, Z, A, activation=activation)
            P1 = forward(W, b1, Z, A, activation=activation)
            J0 = -(np.log(P0)*Y + np.log(1-P0)*(1-Y)).mean()
            J1 = -(np.log(P1)*Y + np.log(1-P1)*(1-Y)).mean()
            rise = J1-J0
            manual_derivative = rise/run
            backprop_derivative = g[frame][ix]
            b_ = np.allclose(manual_derivative,backprop_derivative, rtol=1e-1)
            counter += int(not b_)
    #if any weight/bias was deemed not-close:
    if counter: print("{} out of {} weights are not close".format(counter, n_total_weights))
    return counter


def predict(X, weights, biases, activation=None, probabilities=False):
    #activation function
    d = dict(
    identity = lambda Z : Z,
    relu = lambda Z : np.maximum(0, Z),
    sigmoid = lambda Z : 1 / (1 + np.exp(-Z)),
    tanh = lambda Z : np.tanh(Z)
    )
    activation = d.get(activation, activation)
    
    from types import FunctionType
    if isinstance(activation, type(None)): activation = d["sigmoid"]
    elif isinstance(activation, FunctionType): pass
    else:
        msg = "bad activation function: {}".format(activation)
        raise ValueError(msg)
    

    X = (X-X.mean(0)) / X.std(ddof=0)
    L = len(weights)

    A = X.T
    for l,(W,b) in enumerate(zip(weights,biases)):
        Z = W @ A + b
        A = (d["sigmoid"] if (l==L-1) else activation)(Z)
    p = A.ravel()
    return p if probabilities else (p>=0.5).astype("uint8")



###############################################################################

#make data
activation_function = "tanh"
X,y = make_data(layers=(2,4,3,1), activation=activation_function, seed=None)


"""MLP"""
#hyperparameters
η = 0.5
max_iter = 10000
tol = 0.05
activation = globals().get(activation_function)
do_gradient_check = True

#shapes
m,n = X.shape
layers = (2,4,3,1)
L = len(layers)-1  # number of layers, not counting layer#0
shapes = [(b,a) for a,b in zip(layers[::],layers[1::])]

#initialize
W = [np.ndarray(0)]+[np.random.normal(loc=0, scale=1, size=shape) for shape in shapes]
b = [np.ndarray(0)]+[np.random.normal(loc=0, scale=1, size=(shape[0],1)) for shape in shapes]
n_total_weights = sum(nd.size for nd in W) + sum(nd.size for nd in b)

#make empty collections for future use
Z = [None,]*(L+1)
A = [X.T,] + [None,]*(L)  # X is already scaled
Y = y.reshape(1,-1)
G = [None,] + [None]*L 
g = [None,] + [None]*L 
     
#EPOCHS
for epoch in range(max_iter):
    #forward propagation
    forward(W,b, Z,A, activation=activation)

    #back propagation
    for l in range(L,0,-1):        
        Δ = (W[l+1].T @ Δ) * activation(Z[l], derivative=True) if (l<L) else (A[L]-Y)
        G[l] = (Δ @ A[l-1].T) / m
        g[l] = Δ.sum(axis=1, keepdims=True) / m
    
    #update weights
    for l in range(1, L+1):
        W[l] -= η*G[l]
        b[l] -= η*g[l]
    
    #gradient checking
    if do_gradient_check and not epoch%(max_iter//100):
        gradient_check(W,b, G,g)

    #convergence
    P = A[L]
    J = -(np.log(P)*Y + np.log(1-P)*(1-Y)).mean()
    if not epoch%(max_iter//1000): print(f"cost = {J:.3f}\tepoch {epoch}")
    if J < tol:
        print(f"epoch {epoch}/{max_iter}\tcost = {J:.3f}")
        break
else: print("increase the number of max_iter from", max_iter)

#after epochs
W,b = W[1:], b[1:]
del Z,A,G,g,Δ,P,J


#evaluate
ppred = predict(X, W, b, activation=activation_function, probabilities=True)
ypred = predict(X, W, b, activation=activation_function)
accuracy = np.equal(y,ypred).mean()
print("accuracy =", accuracy.round(3))

#visualize the data
sp = plt.axes(projection="3d")
sp.plot(*X.T, y, '.', markersize=2)
mask = y != ypred
sp.plot(*X[mask].T, y[mask], 'r.', markersize=3)

