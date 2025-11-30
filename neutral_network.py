# Necessary libraries
import numpy
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.experimental import optimizers
from jax import jit 
from jax import grad

# number of nodes in hidden layer
n = 100

def g(u):
    """Activation function. Returns hidden variable"""
    return np.tanh(u)

def V(params:numpy.ndarray, x, y):
    """Function representing the potential. 
    
    Arguments:
        params (np.ndarray): all the weights with length 'n' 
        x (): point on x-axis 
        y (): point on y-axis

    Returns:
        output (float): weighted sum of hidden variables
    """
    w0 = params[:n]
    b0 = params[n:2*n]
    w1 = params[2*n:3*n]
    w2 = params[3*n:4*n]
    b1 = params[4*n]

    hidden = g(x*w0 + y*w1 + b0)
    output = np.sum(hidden*w2) + b1

    return output

# defining the derivatives
dVdx = grad(V, 1)
dVdy = grad(V, 2)
ddVddx = grad(dVdx, 1)
ddVddy = grad(dVdx, 2)
