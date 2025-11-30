# Necessary libraries
import numpy
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.experimental import optimizers
from jax import jit 
from jax import grad
from jax import vmap
from jax import random

# number of nodes in hidden layer
n = 100

def g(u):
    """Activation function. Returns hidden variable"""
    return np.tanh(u)

def V(params:numpy.ndarray, x:float, y:float):
    """Function representing the potential. 
    
    Arguments:
        params (np.ndarray): all the weights with length 'n' 
        x (float): point on x-axis 
        y (float): point on y-axis

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

V_vect = vmap(V, (None, 0, 0))
dVdx_vect = vmap(dVdx, (None, 0, 0))
dVdy_vect = vmap(dVdy, (None, 0, 0))
ddVddx_vect = vmap(ddVddx, (None, 0, 0))
ddVddy_vect = vmap(ddVddy, (None, 0, 0))

# defining initial weights for the neural network
# weights are randomly choosen with normal distribution
key = random.PRNGKey(0)
params = random.normal(key, shape=(4*n+1,))

