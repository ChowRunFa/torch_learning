import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x*w

def cost_function(xs,ys):
    '''
    损失函数：均方误差MSE
    '''
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y-y_pred)**2
    return cost/len(xs)

def gradient_descent(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)
    return grad/len(xs)

print("Predict (before training): )",4,forward(4))

for epoch in range(100):
    cost_value = cost_function(x_data,y_data)
    grad_value = gradient_descent(x_data,y_data)
    w -= 0.01 * grad_value
    print("Epoch:",epoch,"w:",w,"loss:",cost_value)

print("Predict (after training): ",4,forward(4))