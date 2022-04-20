import numpy as np
from GradientDescent import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from sklearn.model_selection import train_test_split # To split train and test models
import sys



# Generate Data
input_var, output_var = generateData()

# Seperate the train and test data
x_train, x_test, y_train, y_test = train_test_split(input_var, output_var, test_size=0.2)

# Initialize weights and bias
w, b = init()


# Train
epoch = 3000
eta = 0.1
accepted_error = 0.05


w, E_train, E_test, min_ep_value = linearRegression(epoch, x_test, y_test, x_train, y_train, w, b, eta, accepted_error)

# Compare Train and test data results
fig2 = plt.figure()

plt.plot(E_train, 'r', label="Train")
plt.plot(E_test, 'b', label="Test")
plt.grid()
plt.legend()
plt.title("Epochs: {}\n Train Error: {:.4f}%\n Test Error {:.4f}%".format(min_ep_value, E_train[min_ep_value], E_test[min_ep_value] ))
plt.show()
    





