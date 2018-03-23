
import numpy as np
import os


def load_mnist_data():
    dirpath=os.path.dirname(__file__)
    fpath = os.path.join(dirpath, 'MNIST-data/mnist.npz')
    f = np.load(fpath)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


