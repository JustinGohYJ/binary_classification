import pandas as pd
import numpy as np
import math
from PIL import Image

EULER = 2.718281828459045

# some safety mechanism
def xavier_init(fan_in, fan_out) -> float:
    val = 6 / (fan_in + fan_out)
    val = math.sqrt(val)
    return val


def sigmoid(val) -> np.array:
    e_to_val = pow(EULER, val)
    fx = e_to_val / (1 + e_to_val)
    return fx


# Return 1d matrix of pixels of size (size x size x 3)
def img_input(path) -> np.arary:
    img = Image.open(path)
    rgb_img = img.convert("RGB")
    
    # convert to r,g,b,r,g,b,r,g,b,... [0-1]
    pixels = np.array(rgb_img).flatten()
    pixels = pixels / 255

    return pixels


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


def forward_propagation(x, w, b):
    Z = []
    A = []

    z = np.dot(x, w[0]) + b[0]
    a = sigmoid(z)

    Z.append(z)
    A.append(a)

    for i in range(1, len(w)):
        z = np.dot(a, w[i]) + b[i]
        a = sigmoid(z)

        Z.append(z)
        A.append(a)

    return Z, A


# binary cross entropy
def obtain_loss(label, pred) -> float:

    # obtain scalar value
    pred = pred.item(0, 0)

    term1 = label * math.log(pred)
    term2 = (1 - label) * math.log(1 - pred)
    loss = -1 * (term1 + term2)
    return loss

X = img_input("pizza_resized/001.png")
W, B = init_network(12288, 4096, 256, 1)

label = True
Z, A = forward_propagation(X, W, B)


loss = obtain_loss(label, A[2])
print(loss)