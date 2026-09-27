import numpy as np
import random
import dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def normalizationFunction(matrix=list):
    matrix = np.array(matrix)
    return (matrix - np.min(matrix))/(np.max(matrix) - np.min(matrix))    

def generateParamters():
    return random.uniform(-0.5, 0.5)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def reLU(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)

def binary_cross_entropy_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def binaryCrossEntropy(actual ,pred):
    actual = np.array(actual)
    pred = np.array(pred)
    loss = -(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))
    return np.mean(loss)

def FeedForward(x, w, b):
    x = np.array(x)
    w = np.array(w)
    b = np.array(b)

    return ( x * w ) + b

def modifiedForward(x1, x2, w1, w2, b):
        x1 = np.array(x1)
        x2 = np.array(x2)

        w1 = np.array(w1)
        w2 = np.array(w2)
        b = np.array(b)
    
        return ( (x1*w1) + (x2*w2) ) + b


def backpropogation():

    global w11, w12, w13, w14, w21, w22, w23, w24, w31, w32, w33, w34, b1, b2, b3, b4, b5
    
    dloss = binary_cross_entropy_derivative(y, a5)
    doutput = dloss * sigmoid_derivative(z5)

    dw31 = doutput * a1
    dw32 = doutput * a2
    dw33 = doutput * a3
    dw34 = doutput * a4

    da1 = doutput * w31 
    dz1 = da1 * relu_derivative(z1)
    da2 = doutput * w32
    dz2 = da2 * relu_derivative(z2)
    da3 = doutput * w33
    dz3 = da3 * relu_derivative(z3)
    da4 = doutput * w34
    dz4 = da4 * relu_derivative(z4)

    dw11 = dz1 * x1
    dw12 = dz2 * x1
    dw13 = dz3 * x1
    dw14 = dz4 * x1

    dw21 = dz1 * x2
    dw22 = dz2 * x2
    dw23 = dz3 * x2
    dw24 = dz4 * x2

    db1 = dz1
    db2 = dz2
    db3 = dz3
    db4 = dz4
    db5 = doutput

    # GRADINT DESCENT 
    lr = 0.001

    w11 -= lr * dw11 
    w12 -= lr * dw12
    w13 -= lr * dw13
    w14 -= lr * dw14

    w21 -= lr * dw21
    w22 -= lr * dw22
    w23 -= lr * dw23
    w24 -= lr * dw24

    w31 -= lr * dw31
    w32 -= lr * dw32
    w33 -= lr * dw33
    w34  -= lr * dw34

    b1 -= lr * db1
    b2 -= lr * db2
    b3 -= lr * db3
    b4 -= lr * db4
    b5 -= lr * db5

    return (
        w11, w12, w13, w14,
        w21, w22, w23, w24,
        w31, w32, w33, w34,
        b1, b2, b3, b4, b5
    )


# INPUTS AND DATA
df = pd.read_csv("Training.csv")
x1 = normalizationFunction(np.array(df["Glucose"]))
x2 = normalizationFunction(np.array(df["Insulin"]))
y = np.array(df.iloc[:, -1])
# WEIGHTS AND BAISES
w11 = generateParamters()
w12 = generateParamters()
w13 = generateParamters()
w14 = generateParamters()

w21 = generateParamters()
w22 = generateParamters()
w23 = generateParamters() 
w24 = generateParamters()

w31 = generateParamters()
w32 = generateParamters()
w33 = generateParamters()
w34 = generateParamters()

b1 = generateParamters()
b2 = generateParamters()
b3 = generateParamters()
b4 = generateParamters()
b5 = generateParamters()

epoch_n = 4000

for epoch in range(0, epoch_n):

    # NUERAL 
    z1 = modifiedForward(x1, x2, w11, w21, b1)
    a1 = reLU(z1)
    z2 = modifiedForward(x1, x2, w12, w22, b2)
    a2 = reLU(z2)
    z3 = modifiedForward(x1, x2, w13, w23, b3)
    a3 = reLU(z3)
    z4 = modifiedForward(x1, x2, w14, w24, b4)
    a4 = reLU(z4)
    z5 = ( (a1*w31) + (a2*w32) + (a3*w33) + (a4*w34) ) + b5
    a5 = sigmoid(z5)

    loss = binaryCrossEntropy(y, a5)


    (
        w11, w12, w13, w14,
        w21, w22, w23, w24,
        w31, w32, w33, w34,
        b1, b2, b3, b4, b5
    ) = backpropogation()


    print(a1[:5])
    print(a2[:5])
    print(a3[:5])
    print(a4[:5])
    print(f"actual : {y[:10]}")
    print(f"output : {np.round(a5[:10], 3)}")
    print(f"loss : {loss}")
    print(f"epoch : {epoch} || loss : {loss}")

