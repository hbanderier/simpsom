import simpsom as sps
from simpsom import SOMNet
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

data = load_digits().data
nx, ny = 10, 10
net = SOMNet(nx, ny, data, init='pca', PBC=True, inner_dist_type='cartesian', neighborhood_fun='mexican_hat')
net.train()
projected = net.project_onto_map(data)
distances = net.neighborhoods.distances
theta = net.theta
a = np.random.randint(nx * ny)
populations = net.compute_populations()

# fig, ax = net.plot_on_map(populations)

residence_times = net.compute_residence_time('max', 1)
# fig, ax = net.plot_on_map(residence_times, 1.0)

trans_mat = net.compute_transmat()
print(net.bmus.shape)
print(trans_mat.shape)
print(np.diag(trans_mat).shape)
fig, ax = net.plot_on_map(np.diag(trans_mat), 1.0)
# plt.show()