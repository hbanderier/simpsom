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
indices = net.find_bmu_ix(data)
populations = np.asarray([(indices == i).sum() for i in range(nx * ny)])
print(indices)
print(indices.shape)
fig, ax = sps.plots.plot_map(
    net.neighborhoods.coordinates,
    theta[56],
    net.polygons,
)
# plt.show()