import numpy as np
import sys

def binary(x):
  return np.heaviside(x,1)

def linear(x):
  return x

def d_linear(x):
  return -x

def sigmoid(x):
  return 1/(1+np.exp(-x))+0.000001

def d_sigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))

def hyperbolic(x):
  return np.tanh(x)

def d_hyperboic(x):
  return 1-hyperbolic(x)**2

def RELU(x):
  return (0 if x<0 else x)

def d_RELU(x):
  return (0 if x<0 else 1)


def diffrentiate(y, y_hat, x_1=1, activation_function = 'linear'):
  if (activation_function == 'linear'):
    d = np.sum(d_linear(y-y_hat) * x_1)
  elif (activation_function == 'sigmoid'):
    d = np.sum(d_sigmoid(y-y_hat) * x_1)
  elif (activation_function == 'hyperbolic'):
    d = np.sum(d_hyperboic(y-y_hat) * x_1)
  elif (activation_function == 'RELU'):
    d = np.sum(d_RELU(y-y_hat) * x_1)
  else: 
    sys.exit('Unknown activation function')
  return d


  
