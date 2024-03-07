import sys
from types import ModuleType
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_distances

import numpy as np
from loguru import logger


class Distance:
    """Container class for distance functions."""

    def __init__(self, xp: ModuleType = None) -> None:
        """Instantiate the Distance class.

        Args:
            xp (numpy or cupy): the numeric labrary to use
                to calculate distances.
        """

        self.xp = xp

    def euclidean_distance(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Calculate Euclidean distance between two arrays.

        Args:
            a (array): first array.
            b (array): second array.

        Returns:
            (float): the manhattan distance
                between two provided arrays.
        """

        if self.xp.__name__ == "cupy":
            _euclidean_distance_kernel = self.xp.ReductionKernel(
                "T x, T w", "T y", "abs(x-w)", "a+b", "y = a", "0", "l2norm"
            )

            d = _euclidean_distance_kernel(
                a[:, None, :],
                b[None, :, :],
                axis=2,
            )
            
            return d

        return pairwise_distances(a, b)

    def cosine_distance(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Calculate the cosine distance between two arrays.

        Args:
            a (array): first array.
            b (array): second array.

        Returns:
            (float): the euclidean distance between two
                provided arrays
        """

        a_sq = self.xp.power(a, 2).sum(axis=1, keepdims=True)

        b_sq = self.xp.power(b, 2).sum(axis=1, keepdims=True)

        similarity = self.xp.nan_to_num(
            self.xp.dot(a, b.T) / self.xp.sqrt(a_sq * b_sq.T)
        )

        return 1 - similarity

    def manhattan_distance(self, a: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Calculate Manhattan distance between two arrays.

        Args:
            a (array): first array.
            b (array): second array.

        Returns:
            (float): the manhattan distance
                between two provided arrays.
        """

        if self.xp.__name__ == "cupy":
            _manhattan_distance_kernel = self.xp.ReductionKernel(
                "T x, T w", "T y", "abs(x-w)", "a+b", "y = a", "0", "l1norm"
            )

            d = _manhattan_distance_kernel(
                a[:, None, :],
                b[None, :, :],
                axis=2,
            )
            
            return d

        return pairwise_distances(a, b, "manhattan")

    def pairdist(self, a: ArrayLike, b: ArrayLike, metric: str) -> ArrayLike:
        """Calculates distances betweens points in batches. Two array-like objects
        must be provided, distances will be calculated between all points in the
        first array and all those in the second array.

        Args:
            x (array): first array.
            w (array): second array.
            metric (string): distance metric.
                Accepted metrics are euclidean, manhattan, and cosine (default "euclidean").
        Returns:
            d (array): the calculated distances.
        """
        a, b = self.xp.atleast_1d(a), self.xp.atleast_1d(b)
        if a.ndim == 1:
            a = a[None, :] # can't use atleast_2d, have to do this
        if b.ndim == 1:
            b = b[None, :] # can't use atleast_2d, have to do this
        if metric == "euclidean":
            return self.euclidean_distance(a, b)

        elif metric == "cosine":
            return self.cosine_distance(a, b)

        elif metric == "manhattan":
            return self.manhattan_distance(a, b)

        logger.error(
            "Available metrics are: " + '"euclidean", "cosine" and "manhattan"'
        )
        sys.exit(1)
