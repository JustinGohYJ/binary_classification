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


def delta_sigmoid(val):
    #element wise multiplication
    return val * (1 - val)


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


# formula recap
# z1 = w1 * x + b1 
# or
# zi = wi * ai-1 + bi
# ai = sig(zi)
# y = label
def back_propagation(x, w, b, z, a, y):
    d_z = []
    d_w = []
    d_b = []

    # last layer delta
    d_last_z = a[-1] - y
    d_last_w = np.dot(a[-2].T, d_last_z)
    d_last_b = d_last_z

    d_z.append(d_last_z)
    d_w.append(d_last_w)
    d_b.append(d_last_b)
    
    # hidden layer delta, not including the first
    for i in range (-2, len(w) * -1, -1):
        print(f"hidden layer: {i}")
        d_i_a = np.dot(d_z[0], w[(i + 1)].T)
        d_i_z = np.multiply(d_i_a, delta_sigmoid(a[i]))

        d_i_w = np.dot(a[i-1].T, d_i_z)
        d_i_b = d_i_z

        d_z.insert(0, d_i_z)
        d_w.insert(0, d_i_w)
        d_b.insert(0, d_i_b)

    # first layer
    d_first_a = np.dot(d_z[0], w[1].T)
    d_first_z = np.multiply(d_first_a, delta_sigmoid(a[0]))

    #reshape input from 1d array to 2d array column vector 
    x = x.reshape(-1, 1)
    d_first_w = np.dot(x, d_first_z)
    d_first_b = d_first_z

    d_w.insert(0, d_first_w)
    d_b.insert(0, d_first_b)

    return d_w, d_b


X = img_input("pizza_resized/001.png")
W, B = init_network(12288, 4096, 256, 128, 1)

label = True
Z, A = forward_propagation(X, W, B)
loss = obtain_loss(label, A[2])
dW, dB = back_propagation(X, W, B, Z, A, label)


for i in range (-1, len(W) * -1 -1, -1):
    print(f"Weights {i}: {np.shape(W[i])}")
    print(f"Bias {i}: {np.shape(B[i])}")
    print(f"Z {i}: {np.shape(Z[i])}")
    print(f"Activation {i}: {np.shape(A[i])}")
    print(f"Dw {i}: {np.shape(dW[i])}")
    print(f"Db {i}: {np.shape(dB[i])}")
    print()
