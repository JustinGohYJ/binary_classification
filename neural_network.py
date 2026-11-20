import pandas as pd
import numpy as np
import math

EULER = 2.718281828459045

# some safety mechanism
def xavier_init(fan_in, fan_out) -> float:
    val = 6 / (fan_in + fan_out)
    val = math.sqrt(val)
    return val

def sigmoid(val) -> float:
    e_to_val = pow(EULER, val)
    fx = e_to_val / (1 + e_to_val)
    return fx

def init_network(*layers):
    '''
    it should return, layers, bias, weights, (and possibly activation)
    the idea is that 
    W is a list of 2d matrix for each layer
    b is a list of 1d matrix for each layer
    '''
    w = []
    b = []
    for layer in range(len(layers) - 1):
        w_val = xavier_init(layers[layer], layers[layer + 1])
        
        w_layer = np.random.uniform(low=(-1 * w_val), 
                                    high=w_val, 
                                    size=(layers[layer], layers[layer + 1]))
        b_layer = np.zeros((1, layers[layer + 1]))

        w.append(w_layer)
        b.append(b_layer)
    return w, b

W, B = init_network(12288, 4096, 256, 256, 1)
        
