import numpy as np
from sklearn.datasets import load_digits
from simpsom import SOMNet
from test_network import TestNetwork

if __name__ == '__main__':
    data = load_digits().data
    net = SOMNet(4, 4, data, init='pca', PBC=True)
    net.train('online')
