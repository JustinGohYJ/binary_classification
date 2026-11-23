# What food is that?

A binary-classification neural network using a multilayer perceptron (MLP) to classify an image as pizza or not pizza.

## Image input and network architecture

A convolutional neural network (CNN) is the more suitable architecture for image data. This project uses a multilayer perceptron (MLP) because it is easier to implement, but using an MLP with the original image size would be impractical.

To make the input size manageable, each image is resized to $64 \\times 64$ pixels. The red, green, and blue values for every pixel are provided to the input layer, giving it $64 \\times 64 \\times 3 = 12{,}288$ input values. The network has two hidden layers, and the estimated training time is around one hour.


## Xavier initialization

Xavier initialization sets each layer's weights randomly within a range scaled by the numbers of input and output neurons, helping keep activations from growing or shrinking too much as they pass through the network. For a layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs, this implementation samples weights uniformly from:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)
$$

Biases are initialized to zero.

## Forward propagation

For the first layer, the input image is transformed by the weights and bias, then passed through the activation function:

$$
Z_1 = XW_1 + b_1
$$

$$
A_1 = \sigma(Z_1)
$$

For each subsequent layer $i$:

$$
Z_i = A_{i-1}W_i + b_i
$$

$$
A_i = \sigma(Z_i)
$$

The activation function is the sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The final output $A_L$ is between 0 and 1. An output greater than 0.5 is classified as pizza; otherwise, it is classified as not pizza.

## Loss

Binary cross-entropy measures the difference between the prediction and the true label:

$$
L = -\left[y\ln(A_L) + (1-y)\ln(1-A_L)\right]
$$

Here, $y=0$ represents not pizza and $y=1$ represents pizza.

## Backpropagation

Backpropagation calculates the gradients of the loss with respect to each weight and bias. For the sigmoid function:

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

### Output-layer gradients

Let $Z_L = A_{L-1}W_L+b_L$ and $A_L=\sigma(Z_L)$. Applying the chain rule to the binary cross-entropy loss gives:

$$
\begin{aligned}
\frac{\partial L}{\partial A_L}
  &= -\left(\frac{y}{A_L}-\frac{1-y}{1-A_L}\right) \\
  &= \frac{A_L-y}{A_L(1-A_L)}
\end{aligned}
$$

The sigmoid derivative at the output is:

$$
\frac{\partial A_L}{\partial Z_L} = A_L(1-A_L)
$$

Therefore, the output-layer error is:

$$
\begin{aligned}
\delta_L
  &= \frac{\partial L}{\partial Z_L} \\
  &= \frac{\partial L}{\partial A_L}\frac{\partial A_L}{\partial Z_L} \\
  &= \frac{A_L-y}{A_L(1-A_L)}A_L(1-A_L) \\
  &= A_L-y
\end{aligned}
$$

Since $Z_L=A_{L-1}W_L+b_L$, the weight gradient is:

$$
\begin{aligned}
\frac{\partial L}{\partial W_L}
  &= \left(\frac{\partial L}{\partial Z_L}\right)\left(\frac{\partial Z_L}{\partial W_L}\right) \\
  &= A_{L-1}^{T}\delta_L \\
  &= A_{L-1}^{T}(A_L-y)
\end{aligned}
$$

The bias gradient follows because $\partial Z_L/\partial b_L=1$:

$$
\begin{aligned}
\frac{\partial L}{\partial b_L}
  &= \frac{\partial L}{\partial Z_L}\frac{\partial Z_L}{\partial b_L} \\
  &= \delta_L \cdot 1 \\
  &= A_L-y
\end{aligned}
$$

### Hidden-layer gradients

For one hidden layer, let $Z_1=XW_1+b_1$ and $A_1=\sigma(Z_1)$. The error is propagated back from the next layer by the chain rule:

$$
\begin{aligned}
\frac{\partial L}{\partial A_1}
  &= \frac{\partial L}{\partial Z_2}\frac{\partial Z_2}{\partial A_1} \\
  &= \delta_2 W_2^{T}
\end{aligned}
$$

Applying the sigmoid derivative elementwise gives the hidden-layer error:

$$
\begin{aligned}
\delta_1
  &= \frac{\partial L}{\partial Z_1} \\
  &= \frac{\partial L}{\partial A_1}\odot\frac{\partial A_1}{\partial Z_1} \\
  &= (\delta_2 W_2^{T})\odot A_1(1-A_1)
\end{aligned}
$$

Here, $\delta_2$ is the error term from the following layer, and $\odot$ denotes elementwise multiplication. The hidden-layer weight gradient then follows from $Z_1=XW_1+b_1$:

$$
\begin{aligned}
\frac{\partial L}{\partial W_1}
  &= \left(\frac{\partial L}{\partial Z_1}\right)\left(\frac{\partial Z_1}{\partial W_1}\right) \\
  &= X^{T}\delta_1
\end{aligned}
$$

The hidden-layer bias gradient is:

$$
\begin{aligned}
\frac{\partial L}{\partial b_1}
  &= \frac{\partial L}{\partial Z_1}\frac{\partial Z_1}{\partial b_1} \\
  &= \delta_1\cdot 1 \\
  &= \delta_1
\end{aligned}
$$

## Parameter updates

Weights and biases are updated using gradient descent:

$$
p_{\text{new}} = p_{\text{old}} - \alpha\frac{\partial L}{\partial p}
$$

Here, $p$ denotes a weight or bias, and $\alpha$ is the learning rate.