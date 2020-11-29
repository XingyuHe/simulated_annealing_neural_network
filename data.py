import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from model import *

def load_data():
    file_name = "./mnist.npz"
    if os.path.exists(file_name):
        f = np.load(file_name)

        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def plot(x):

    plt.plot(x)
    plt.show()