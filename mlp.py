import random
from pathlib import Path

import numpy as np
import math
import data
from PIL import Image

EULER = 2.718281828459045

# for sigmoid
def xavier_init(fan_in, fan_out) -> float:
    val = 6 / (fan_in + fan_out)
    val = math.sqrt(val)
    return val


# for relu
def he_init(fan_in) -> float:
    return math.sqrt(6 / fan_in)


# mathematical equivalent
def sigmoid(val) -> np.array:
    return (0.5 * (1 + np.tanh(0.5 * val)))


def delta_sigmoid(val) -> np.array:
    #element wise multiplication
    return val * (1 - val)


def relu(val) -> np.array:
    return np.maximum(0, val)


def delta_relu(val) -> np.array:
    return val > 0


# Return 1d matrix of pixels of size (size x size x 3)
def img_input(path) -> np.arary:
    img = Image.open(path)
    rgb_img = img.convert("RGB")
    
    # convert to r,g,b,r,g,b,r,g,b,... [0-1]
    pixels = np.array(rgb_img).flatten()
    pixels = pixels / 255

    return pixels

# he init for all except last
def init_network(*layers):
    '''
    W is a list of 2d matrix for each layer
    b is a list of 1d matrix for each layer
    '''
    w = []
    b = []
    for layer in range(len(layers) - 1):
        if layer == len(layers) - 2: #last layer
            w_val = xavier_init(layers[layer], layers[layer + 1])
        else:
            w_val = he_init(layers[layer])
        
        w_layer = np.random.uniform(low=(-1 * w_val), 
                                    high=w_val, 
                                    size=(layers[layer], layers[layer + 1]))
        b_layer = np.zeros((1, layers[layer + 1]))

        w.append(w_layer)
        b.append(b_layer)
    return w, b


# applies relu to all except last
def forward_propagation(x, w, b):
    Z = []
    A = []

    for i in range(len(w)): 
        if i == 0: 
            prev_a = x
        else:
            prev_a = A[-1]
        
        z = np.dot(prev_a, w[i]) + b[i]
        if i == len(w) - 1:
            a = sigmoid(z) 
        else:
            a = relu(z)
        
        Z.append(z)
        A.append(a)

    return Z, A


# binary cross entropy
def obtain_loss(label, pred) -> float:
    # prevent breaking log function at 0
    pred = np.clip(pred.item(0, 0), 1e-15, 1 - 1e-15)

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
        d_i_a = np.dot(d_z[0], w[(i + 1)].T)
        d_i_z = np.multiply(d_i_a, delta_relu(z[i]))

        d_i_w = np.dot(a[i-1].T, d_i_z)
        d_i_b = d_i_z

        d_z.insert(0, d_i_z)
        d_w.insert(0, d_i_w)
        d_b.insert(0, d_i_b)

    # first layer
    d_first_a = np.dot(d_z[0], w[1].T)
    d_first_z = np.multiply(d_first_a, delta_relu(z[0]))

    #reshape input from 1d array to 2d array column vector 
    x = x.reshape(-1, 1)
    d_first_w = np.dot(x, d_first_z)
    d_first_b = d_first_z

    d_w.insert(0, d_first_w)
    d_b.insert(0, d_first_b)

    return d_w, d_b


# not used
def update_param(dw, db, w, b, learning_rate):
    for i in range(len(w)):
        w[i] -= learning_rate * dw[i]
        b[i] -= learning_rate * db[i]

    return w, b


class Momentum:
    def __init__(self, w, b, beta=0.9):
        self.beta = beta
        self.m_w = [np.zeros_like(wi) for wi in w]
        self.m_b = [np.zeros_like(bi) for bi in b]

    # heavy ball alg
    def update(self, dw, db, w, b, learning_rate):
        for i in range(len(w)):
            self.m_w[i] = self.beta * self.m_w[i] + dw[i]
            w[i] -= learning_rate * self.m_w[i]

            self.m_b[i] = self.beta * self.m_b[i] + db[i]
            b[i] -= learning_rate * self.m_b[i]

        return w, b


# training loss calculated after
def full_pass(train, w, b, learning_rate, batch_size, optimizer):
    
    random.shuffle(train)

    train_amt = len(train)
    total_batches = math.ceil(train_amt / batch_size)

    # enumerate just for print
    for batch_number, start in enumerate(range(0, train_amt, batch_size)):
        batch = train[start:(start + batch_size)]
        
        w, b = run_mini_batch(
            batch, w, b, learning_rate, optimizer
        )

        print(
            f"\rMini-batch {batch_number + 1}/{total_batches}",
            end="",
            flush=True,
        )
    print()
    return w, b


def run_mini_batch(batch, w, b, learning_rate, optimizer):
    mlp_size = len(w)
    batch_dw = [np.zeros_like(layer) for layer in w]
    batch_db = [np.zeros_like(layer) for layer in b]
    batch_loss = 0.0

    # for each image
    for path, label in batch:
        x = img_input(path)
        z, a = forward_propagation(x, w, b)
        batch_loss += obtain_loss(label, a[-1])
        dw, db = back_propagation(x, w, b, z, a, label)

        for layer in range(mlp_size):
            batch_dw[layer] += dw[layer]
            batch_db[layer] += db[layer]

    # last batch can be smaller, used len()
    for layer in range(mlp_size):
        batch_dw[layer] /= len(batch)
        batch_db[layer] /= len(batch)

    w, b = optimizer.update(batch_dw, batch_db, w, b, learning_rate)
    return w, b


# evaluate average loss and accuracy
def evaluate(samples, w, b):
    amt = len(samples)
    total_loss = 0.0
    
    labels = []
    predictions = []

    for path, label in samples:
        x = img_input(path)
        z, a = forward_propagation(x, w, b)
        prediction = a[-1].item(0, 0)
        total_loss += obtain_loss(label, a[-1])
        
        labels.append(label)
        predictions.append(prediction)

    avg_loss = total_loss / amt
    accuracy = obtain_accuracy(labels, predictions)
    return avg_loss, accuracy


# just check bool with the label 
# 0 is non pizza, 1 is pizza
def obtain_accuracy(labels, predictions):
    correct = 0

    for label, prediction in zip(labels, predictions):
        guess = int(prediction >= 0.5)
        correct += (guess == label)

    return correct / len(labels)