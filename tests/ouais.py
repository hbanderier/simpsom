from simpsom import SOMNet
from sklearn.datasets import load_digits

data = load_digits().data
net = SOMNet(10, 10, data, init='pca').train()