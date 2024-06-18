import multiprocessing
import os
import sys
from functools import partial
from typing import Union, List, Tuple, Callable
from collections.abc import Iterable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from nptyping import NDArray

import numpy as np
from scipy.stats import linregress
from loguru import logger
from tqdm import trange

from simpsom.distances import *
from simpsom.neighborhoods import Neighborhoods
from simpsom.plots import plot_map
from sklearn.metrics import pairwise_distances
logger.add(sys.stderr, level="ERROR")

class SOMNet:
    """Kohonen SOM Network class."""

    def __init__(
        self,
        net_width: int,
        net_height: int,
        data: np.ndarray,
        load_file: str = None,
        metric: str = "ssim",
        topology: str = "hexagonal",
        inner_dist_type: str | Iterable[str] = "grid",
        neighborhood_fun: str = "gaussian",
        init: str = "random",
        PBC: bool = False,
        random_seed: int = None,
        debug: bool = False,
        output_path: str = "./",
    ) -> None:
        """Initialize the SOM network.

        Args:
            net_height (int): Number of nodes along the first dimension.
            net_width (int): Numer of nodes along the second dimension.
            data (array): N-dimensional dataset.
            load_file (str): Name of file to load containing information
                to initialize the network weights.
            metric (string): distance metric for the identification of best matching
                units. Accepted metrics are euclidean, manhattan, and cosine (default "euclidean").
            topology (str): topology of the map tiling.
                Accepted shapes are hexagonal, and square (default "hexagonal").
            neighborhood_fun (str): neighbours drop-off function for training, choose among gaussian,
                mexican_hat and bubble (default "gaussian").
            init (str or list[array, ...]): Nodes initialization method, choose between random
                or PCA (default "random").
            PBC (boolean): Activate/deactivate periodic boundary conditions,
                warning: only quality threshold clustering algorithm works with PBC (default False).
            GPU (boolean): Activate/deactivate GPU run with RAPIDS (requires CUDA, default False).
            CUML (boolean): Use CUML for clustering. If deactivate, use scikit-learn instead
                (requires CUDA, default False).
            random_seed (int): Seed for the random numbers generator (default None).
            debug (bool): Set logging level printed to screen as debug.
            out_path (str): Path to the folder where all data and plots will be saved
                (default, current folder).
        """

        self.output_path = output_path
        self.data_range = 1.
        if not debug:
            logger.remove()
            logger.add(sys.stderr, level="ERROR")

        if random_seed is not None:
            os.environ["PYTHONHASHSEED"] = str(random_seed)
            np.random.seed(random_seed)

        self.PBC = bool(PBC)
        if self.PBC:
            logger.info("Periodic Boundary Conditions active.")

        self.width = net_width
        self.height = net_height
        self.n_nodes = self.width * self.height

        self.data = np.array(data, dtype=np.float32)

        self.metric = metric

        if topology.lower() == "hexagonal":
            logger.info("Hexagonal topology.")
            self.polygons = "Hexagons"
        else:
            self.polygons = "Squares"
            logger.info("Square topology.")

        self.inner_dist_type = inner_dist_type

        self.neighborhood_fun = neighborhood_fun.lower()
        if self.neighborhood_fun not in ["gaussian", "mexican_hat", "bubble"]:
            logger.error(
                "{} neighborhood function not recognized.".format(self.neighborhood_fun)
                + "Choose among 'gaussian', 'mexican_hat' or 'bubble'."
            )
            raise ValueError

        self.neighborhoods = Neighborhoods(
            self.width,
            self.height,
            self.polygons,
            self.inner_dist_type,
            self.PBC,
        )

        self.convergence = []

        self.init = init
        if isinstance(self.init, str):
            self.init = self.init.lower()
        else:
            self.init = np.array(self.init)
        self._set_weights(load_file)
        if self.metric.lower() == "ssim":
            self.data_range = data.max() - data.min()
        

    def _set_weights(self, load_file: str = None) -> None:
        """Set initial map weights values, either by loading them from file or with random/PCA.

        Args:
            load_file (str): Name of file to load containing information
                to initialize the network weights.
        """

        init_vec = None

        # When loaded from file, element 0 contains information on the network shape

        if load_file is None:
            if isinstance(self.init, str) and self.init == "pca":
                logger.warning(
                    "Please be sure that the data have been standardized before using PCA."
                )
                logger.info("The weights will be initialized with PCA.")
                matrix = self.data
                init_vec = self.pca(matrix, n_pca=2)

            else:
                logger.info("The weights will be initialized randomly.")
                init_vec = [
                    np.min(self.data, axis=0),
                    np.max(self.data, axis=0),
                ]
            self.weights = (
                init_vec[0][None, :]
                + (init_vec[1] - init_vec[0])[None, :]
                * np.random.rand(self.n_nodes, *init_vec[0].shape)
            ).astype(np.float32)

        else:
            logger.info("The weights will be loaded from file")
            if not load_file.endswith(".npy"):
                load_file += ".npy"
            self.weights = np.load(load_file, allow_pickle=True)
            self.bmus = self.find_bmu_ix(self.data)
            neighborhood_caller = partial(
                self.neighborhoods.neighborhood_caller,
                neigh_func=self.neighborhood_fun,
            )
            self.sigma = 0.1

    def pca(self, matrix: np.ndarray, n_pca: int) -> np.ndarray:
        """Get principal components to initialize network weights.

        This is a numpy-only function until cupy implements linalg.eig.

        Args:
            matrix (array): N-dimensional dataset.
            n_pca (int): number of components to keep.

        Returns:
            (array): Principal axes in feature space,
                representing the directions of maximum variance in the data.
        """

        mean_vector = np.mean(matrix.T, axis=1)
        center_mat = matrix - mean_vector
        logger.info('cov_mat')
        cov_mat = np.cov(center_mat.T).astype(np.float32)
        logger.info('linalg.eig')
        return np.linalg.eig(cov_mat)[-1].T[:n_pca]

    def _randomize_dataset(self, data: np.ndarray, epochs: int) -> np.ndarray:
        """Generates a random list of datapoints indices for online training.

        Args:
            data (array or list): N-dimensional dataset.
            epochs (int): Number of training iterations.

        Returns:
            entries (array): array with randomized indices
        """

        if epochs < data.shape[0]:
            logger.warning(
                "Epochs for online training are less than the input datapoints."
            )
            epochs = data.shape[0]

        iterations = int(np.ceil(epochs / data.shape[0]))

        return [
            ix
            for shuffled in [
                np.random.permutation(data.shape[0]) for _ in np.arange(iterations)
            ]
            for ix in shuffled
        ]

    def save_map(self, file_name: str = "trained_som.npy") -> None:
        """Saves the network dimensions, the pbc and nodes weights to a file.

        Args:
            file_name (str): Name of the file where the data will be saved.
        """

        if not file_name.endswith((".npy")):
            file_name += ".npy"
        logger.info(
            "Map shape and weights will be saved to:\n"
            + os.path.join(self.output_path, file_name)
        )
        np.save(os.path.join(self.output_path, file_name), self.weights)

    def _update_sigma(self, epoch: int) -> None:
        """Update the gaussian sigma.

        Args:
            n_iter (int): Iteration number.
        """

        self.sigma = self.start_sigma * np.exp(epoch / (self.epochs - 1) * np.log(self.end_sigma / self.start_sigma))

    def _update_learning_rate(self, epoch: int) -> None:
        """Update the learning rate.

        Args:
            n_iter (int): Iteration number.
        """

        self.learning_rate = self.start_learning_rate * np.exp(-epoch / self.epochs)
        # self.learning_rate = self.start_learning_rate

    def find_bmu_ix(self, vecs: NDArray) -> int:
        """Find the index of the best matching unit (BMU) for a given list of vectors.

        Args:
            vec (array or list[lists, ..]): vectors whose distance from the network
                nodes will be calculated.

        Returns:
            bmu (int): The best matching unit node index.
        """
        if self.metric.lower() == "euclidean":
            dists = pairwise_distances(
                vecs.reshape(vecs.shape[0], -1),
                self.weights.reshape(self.weights.shape[0], -1),
                n_jobs=12,
            )
        elif self.metric.lower() == "ssim":
            dists = pairwise_ssim(
                vecs,
                self.weights,
                win_size=21,
                strides=3,
                data_range=self.data_range
            )
        return np.argmin(dists, axis=1)
    
    def online_train(self, neighborhood_caller: Callable) -> None:
        datapoints_ix = self._randomize_dataset(self.data, self.epochs)

        for epoch in trange(self.epochs):
            self._update_sigma(epoch)
            self._update_learning_rate(epoch)

            datapoint_ix = datapoints_ix.pop()
            input_vec = self.data[datapoint_ix]

            bmu = int(self.find_bmu_ix(input_vec)[0])

            self.theta = (
                neighborhood_caller(bmu, sigma=self.sigma) * self.learning_rate
            )

            self.weights -= self.theta[:, None] * (
                self.weights - input_vec[None, :]
            )
            
    def batch_train(self, neighborhood_caller: Callable) -> None:
        nbatch = len(self.data) // 256
        nodes = np.arange(self.n_nodes)
        pre_numerator = np.zeros(self.weights.shape, dtype=np.float32)
        numerator = pre_numerator.copy()
        for epoch in (pbar := trange(self.epochs)):
            batches = np.array_split(self.data, nbatch)
            batches = [batches[i] for i in np.random.choice(nbatch, nbatch, replace=False)]
            self._update_sigma(epoch)
            self._update_learning_rate(epoch)
            h = neighborhood_caller(nodes, sigma=self.sigma)
            
            for batch in batches:
                # Find BMUs for all points and subselect gaussian matrix.
                indices = self.find_bmu_ix(batch)

                series = indices[:, None] == nodes[None, :]
                pop = np.sum(series, axis=0, dtype=np.float32)
                for i, s in enumerate(series.T):
                    pre_numerator[i, :, :] = np.sum(batch[s], axis=0)
                numerator = np.einsum('ij, jkl -> ikl', h, pre_numerator)
                    
                denominator = (h @ pop)[:, None, None]

                new_weights = np.where(
                    denominator != 0, numerator / denominator, self.weights
                )

                self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights
                self.weights = self.weights.astype(np.float32)
            pbar.set_description(f'lr: {self.learning_rate:.2e}, sigma: {self.sigma:.2e}')
                
    def train(
        self,
        train_algo: str = "batch",
        epochs: int = -1,
        start_learning_rate: float = 0.05,
    ) -> None:
        """Train the SOM.

        Args:
            train_algo (str): training algorithm, choose between "online" or "batch"
                (default "online"). Beware that the online algorithm will run one datapoint
                per epoch, while the batch algorithm runs all points at one for each epoch.
            epochs (int): Number of training iterations. If not selected (or -1)
                automatically set epochs as 10 times the number of datapoints.
                Warning: for online training each epoch corresponds to 1 sample in the
                input dataset, for batch training it corresponds to one full dataset
                training.
            start_learning_rate (float): Initial learning rate, used only in online
                learning.
            early_stop (str): Early stopping method, for now only "mapdiff" (checks if the
                weights of nodes don"t change) is available. If None, don"t use early stopping (default None).
            early_stop_patience (int): Number of iterations without improvement before stopping the
                training, only available for batch training (default 3).
            early_stop_tolerance (float): Improvement tolerance, if the map does not improve beyond
                this threshold, the early stopping counter will be activated (it needs to be set
                appropriately depending on the used distance metric). Ignored if early stopping
                is off (default 1e-4).
        """

        logger.info("The map will be trained with the " + train_algo + " algorithm.")
        self.start_sigma = min(self.width, self.height) / 2
        self.start_learning_rate = start_learning_rate

        self.data = np.array(self.data)

        if epochs == -1:
            if train_algo == "online":
                epochs = self.data.shape[0] * 10
            else:
                epochs = 100

        self.epochs = epochs
        self.end_sigma = .1

        neighborhood_caller = partial(
            self.neighborhoods.neighborhood_caller,
            neigh_func=self.neighborhood_fun,
        )

        if train_algo == "online":
            """Online training.
            Bootstrap: one datapoint is extracted randomly with replacement at each epoch
            and used to update the weights.
            """

            self.online_train(neighborhood_caller)

        elif train_algo == "batch":
            """Batch training.
            All datapoints are used at once for each epoch,
            the weights are updated with the sum of contributions from all these points.
            No learning rate needed.

            Kinouchi, M. et al. "Quick Learning for Batch-Learning Self-Organizing Map" (2002).
            """
            self.batch_train(neighborhood_caller)

        self.bmus = self.find_bmu_ix(self.data)

    def get_nodes_difference(self) -> None:
        """Extracts the neighbouring nodes difference in weights and assigns it
        to each node object.
        """

        weights_dist = self.distance.pairdist(
            self.weights, self.weights, metric=self.metric
        )
        pos_dist = self.neighborhoods.distances

        weights_dist[(pos_dist > 1.01) | (pos_dist == 0.0)] = np.nan
        self.differences = np.nanmean(weights_dist, axis=0)

        logger.info("Weights difference among neighboring nodes calculated.")

    def project_onto_map(
        self, array: NDArray = None, file_name: str = "./som_projected.npy"
    ) -> NDArray:
        """Project the datapoints of a given array to the 2D space of the
        SOM by calculating the bmus.

        Args:
            array (array): An array containing datapoints to be mapped.
            file_name (str): Name of the file to which the data will be saved
                if not None.

        Returns:
            (list): bmu x,y position for each input array datapoint.
        """
        if array is None:
            array = self.data
        elif not isinstance(array, np.ndarray):
            array = np.array(array)

        bmu_coords = self.neighborhoods.coordinates[self.find_bmu_ix(array)]

        if file_name is not None:
            if not file_name.endswith((".npy")):
                file_name += ".npy"
            logger.info(
                "Projected coordinates will be saved to:\n"
                + os.path.join(self.output_path, file_name)
            )
            np.save(os.path.join(self.output_path, file_name), bmu_coords)

        return np.array(bmu_coords)

    def compute_populations(self, data: NDArray = None) -> NDArray:
        if data is None:
            indices = self.bmus
        else:
            indices = self.find_bmu_ix(data)
        return np.asarray([np.sum(indices == i) for i in range(self.n_nodes)])

    def compute_transmat(self, data: NDArray = None, step: int = 1, yearbreaks: int = 92) -> NDArray:
        if data is None:
            indices = self.bmus
        else:
            indices = self.find_bmu_ix(data)

        trans_mat = np.zeros((self.n_nodes, self.n_nodes))
        start_point = 0
        for end_point in range(yearbreaks, len(indices), yearbreaks): # cleaner version with slices instead of fixed length summer if I ever need to to it for winter ? Flemme
            real_end_point = min(end_point, len(indices) - 1)
            theseind = np.vstack(
                [indices[start_point:real_end_point-step], np.roll(indices[start_point:real_end_point], -step
            )[:-step]]).T
            theseind, counts = np.unique(theseind, return_counts=True, axis=0)
            trans_mat[theseind[:, 0], theseind[:, 1]] += counts
            start_point = real_end_point
        trans_mat /= np.sum(trans_mat, axis=1)[:, None]
        return np.nan_to_num(trans_mat, nan=0)

    def compute_residence_time(
        self, smooth_sigma: float = 0.0, yearbreak: int = 92,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        all_lengths = []
        all_lenghts_flat = []
        for j in range(self.n_nodes):
            all_lengths.append([])
            all_lenghts_flat.append([])
        start_point = 0
        distances = self.neighborhoods.distances
        indices = self.bmus
        for end_point in range(yearbreak, len(indices) + 1, yearbreak):
            for j in range(self.n_nodes):
                all_lengths[j].append([0])
            real_end_point = min(end_point, len(indices) - 1)
            these_indices = indices[start_point:real_end_point]
            jumps = np.where(distances[these_indices[:-1], these_indices[1:]] != 0)[0]
            beginnings = np.append([0], jumps + 1)
            lengths = np.diff(np.append(beginnings, [yearbreak]))
            if smooth_sigma != 0:
                series_distances = (distances[these_indices[beginnings], :][:, these_indices[beginnings]] <= smooth_sigma).astype(int)
                series_distances[np.tril_indices_from(series_distances, k=-1)] = 0
                how_many_more = np.argmax(np.diff(series_distances, axis=1) == -1, axis=1)[:-1] - np.arange(len(beginnings) - 1)
                for i in range(len(lengths) - 1):
                    lengths[i] = np.sum(lengths[i:i + how_many_more[i] + 1])
            for node, length in zip(these_indices[beginnings], lengths):
                all_lengths[node][-1].append(length)
                all_lenghts_flat[node].append(length)
            start_point = real_end_point
        trend_lengths = []
        max_lengths = []
        mean_lengths = []
        pvalues = []
        for i in range(self.n_nodes):
            mean_lengths.append(np.mean(all_lenghts_flat[i]))
            max_each_year = np.asarray([np.amax(all_lengths[i][j]) for j in range(len(all_lengths[i]))])
            max_lengths.append(np.amax(max_each_year))
            mask = max_each_year != 0
            trend, _, _, pvalue, _ = linregress(np.arange(len(all_lengths[i]))[mask], max_each_year[mask])
            trend_lengths.append(trend)
            pvalues.append(pvalue)
        mean_lengths = np.asarray(mean_lengths)
        max_lengths = np.asarray(max_lengths)
        trend_lengths = np.asarray(trend_lengths)
        pvalues = np.asarray(pvalues)
        return mean_lengths, max_lengths, trend_lengths, pvalues

    def compute_autocorrelation(
        self, data: NDArray = None, lag_max: int = 50
    ) -> NDArray:
        if data is None:
            indices = self.bmus
        else:
            indices = self.find_bmu_ix(data)
        series = indices[None, :] == np.arange(self.n_nodes)[:, None]
        autocorrs = []
        for i in range(lag_max):
            autocorrs.append(
                np.diag(
                    np.corrcoef(series[:, i:], np.roll(series, i, axis=1)[:, i:])[
                        : self.n_nodes, self.n_nodes :
                    ]
                )
            )
        return np.asarray(autocorrs)

    def smooth(
        self,
        data: NDArray,
        smooth_sigma: float = 0,
        neigh_func: str = None,
    ) -> NDArray:
        if np.isclose(smooth_sigma, 0.0):
            return data
        if neigh_func is None:
            neigh_func = self.neighborhood_fun
        theta = self.neighborhoods.neighborhood_caller(
            np.arange(self.n_nodes), smooth_sigma, neigh_func=neigh_func
        )
        return np.sum((data[None, :] * theta), axis=1) / np.sum(theta, axis=1)

    def plot_on_map(
        self,
        data: NDArray,
        smooth_sigma: float = 0,
        fig: Figure = None,
        ax: Axes = None,
        draw_cbar: bool = True,
        **kwargs: Tuple
    ) -> Tuple[Figure, Axes]:
        """Wrapper function to plot a trained 2D SOM map
        color-coded according to a given feature.

        Args:
            data (NDArray): What to show on the map.
            show (bool): Choose to display the plot.
            print_out (bool): Choose to save the plot to a file.
            kwargs (dict): Keyword arguments to format the plot, such as
                - figsize (tuple(int, int)), the figure size;
                - title (str), figure title;
                - cbar_label (str), colorbar label;
                - labelsize (int), font size of label, the title 15% larger, ticks 15% smaller;
                - cmap (ListedColormap), a custom cmap.
        """

        data = self.smooth(data, smooth_sigma)
        fig, ax = plot_map(
            self.neighborhoods.coordinates,
            data,
            self.polygons,
            fig=fig,
            ax=ax,
            draw_cbar=draw_cbar,
            **kwargs
        )

        return fig, ax
