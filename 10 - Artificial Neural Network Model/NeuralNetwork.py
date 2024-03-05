import numpy as np

# Inputs [sleep, study] and output [Expected% in Exams]
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / 100


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def derivatives_sigmoid(x):
    return x * (1 - x)


# Variable initialization
epoch = 5000  # Training iterations
lr = 0.1  # Learning rate
inputlayer_neurons = 2  # Number of features in dataset
hiddenlayer_neurons = 3  # Number of hidden layers neurons
output_neurons = 1  # Number of neurons at output layer

# Weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training algorithm
for i in range(epoch):
    # Forward propagation
    hin = np.dot(X, wh) + bh
    h_layer_act = sigmoid(hin)
    out_hin = np.dot(h_layer_act, wout) + bout
    output = sigmoid(out_hin)

    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(h_layer_act)
    d_hiddenlayer = EH * hiddengrad

    wout += h_layer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

# Output
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)
