import numpy as np
from ActivationFunction import *
from cmath import inf
import matplotlib.pyplot as plt
import pandas as pd



def init():
    # Initialize weights and bias
    w = np.random.random_sample(2)
    b = 1.0
    return w, b

def generateData(data="random"):
    # if data=="random":
    """ Generate data """
    point_count = 100
    rand_coeff = 50.0

    slope = np.random.randint(-5, 5)
    intercept = np.random.randint(-5, 5)
    input_var = np.random.rand(point_count)
    output_var = (slope*input_var + intercept)+ np.random.rand(point_count)
    return input_var, output_var
        




def MSE(Loss, N):
    return 0.5*np.sum(Loss)/N


def linearRegression(epoch, x_test, y_test, x_train, y_train, w, b, eta, accepted_error):
    m_train = x_train.size
    m_test = x_test.size

    total_err = MSE(x_train, m_train)
    min_ep_value = 0
    percent_error = 0
    E_train = []
    E_test = []
    percent_error_old = inf

    """To plot the data"""
    fig = plt.figure()

    for i in range(epoch):
        Z_train = w[1]*x_train + w[0]*b
        Z_test = w[1]*x_test + w[0]*b

        y_hat_train = linear(Z_train)
        y_hat_test = linear(Z_test)

        L_train = (y_train-y_hat_train)**2
        L_test = (y_test-y_hat_test)**2

        error_train = MSE(L_train, m_train)
        error_test = MSE(L_test, m_test)

        E_train.append(error_train)
        E_test.append(error_test)
        # Backpropagation
        dEdw_1 = diffrentiate(y_train, y_hat_train, x_train, activation_function="linear")
        dEdw_0 = diffrentiate(y_train, y_hat_train, activation_function="linear")

        # Uppdate weights
        w[1] -= eta*dEdw_1/m_train
        w[0] -= eta*dEdw_0/m_train

        
        percent_error = (error_train/total_err)*100
        # Show the progress every 50 epochs
        if i % 50 == 0 or percent_error <= accepted_error or abs(percent_error_old - percent_error) < 1e-6:
            plt.scatter(x_train, y_train, c='r', alpha=0.5)
            #plot the prediction line
            plt.plot(x_train, y_hat_train, '-b', label="MSE: {:.4f}%\nEpochs: {}".format(percent_error, i))
            plt.title("Linear Regression")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid()
            plt.show()
        if i > min_ep_value:
            min_ep_value = i
        if percent_error <= accepted_error or abs(percent_error_old - percent_error) < 1e-2:
            break        
        percent_error_old = percent_error

    return w, E_train, E_test, min_ep_value


