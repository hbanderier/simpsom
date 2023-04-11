import sys
from types import ModuleType
from numpy.typing import ArrayLike

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

    def euclidean_distance(self, x: ArrayLike, w: ArrayLike) -> ArrayLike:
        """Calculate the L2 distance between two arrays.

        Args:
            x (array): first array.
            w (array): second array.

        Returns:
            (float): the euclidean distance between two
                provided arrays
        """

        return self.xp.sqrt(self.xp.power(x[:, None, :] - w[None, :, :], 2).sum(axis=2))


    def cosine_distance(self, x: ArrayLike, w: ArrayLike) -> ArrayLike:
        """Calculate the cosine distance between two arrays.

        Args:
            x (array): first array.
            w (array): second array.

        Returns:
            (float): the euclidean distance between two
                provided arrays
        """

        x_sq = self.xp.power(x, 2).sum(axis=1, keepdims=True)

        w_sq = self.xp.power(w, 2).sum(axis=1, keepdims=True)

        similarity = self.xp.nan_to_num(
            self.xp.dot(x, w.T) / self.xp.sqrt(x_sq * w_sq.T)
        )

        return 1 - similarity

    def manhattan_distance(self, x: ArrayLike, w: ArrayLike) -> ArrayLike:
        """Calculate Manhattan distance between two arrays.

        Args:
            x (array): first array.
            w (array): second array.

        Returns:
            (float): the manhattan distance
                between two provided arrays.
        """

        if self.xp.__name__ == "cupy":
            _manhattan_distance_kernel = self.xp.ReductionKernel(
                "T x, T w", "T y", "abs(x-w)", "a+b", "y = a", "0", "l1norm"
            )

            d = _manhattan_distance_kernel(
                x[:, None, :],
                w[None, :, :],
                axis=2,
            )

        else:
            d = self.xp.linalg.norm(
                x[:, None, :] - w[None, :, :],
                ord=1,
                axis=2,
            )

        return d

    def batchpairdist(self, x: ArrayLike, w: ArrayLike, metric: str) -> ArrayLike:
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

        if metric == "euclidean":
            return self.euclidean_distance(x, w)

        elif metric == "cosine":
            return self.cosine_distance(x, w)

        elif metric == "manhattan":
            return self.manhattan_distance(x, w)

        logger.error(
            "Available metrics are: " + '"euclidean", "cosine" and "manhattan"'
        )
        sys.exit(1)

    def pairdist(self, a: ArrayLike, b: ArrayLike, metric: str) -> ArrayLike:
        """Calculates distances betweens points. Two array-like objects
        must be provided, distances will be calculated between all points in the
        first array and all those in the second array.

        Args:
            a (array): first array.
            b (array): second array.
            metric (string): distance metric.
                Accepted metrics are euclidean, manhattan, and cosine (default "euclidean").

        Returns:
            d (array or list): the calculated distances.
        """
        a, b = self.xp.atleast_1d(a), self.xp.atleast_1d(b)
        if a.ndim == 1:
            a = a[None, :] # can't use atleast_2d, have to do this
        if b.ndim == 1:
            b = b[None, :] # can't use atleast_2d, have to do this
        if metric == "euclidean":
            squares_a = self.xp.sum(self.xp.power(a, 2), axis=1, keepdims=True)
            squares_b = self.xp.sum(self.xp.power(b, 2), axis=1, keepdims=True)
            return self.xp.sqrt(squares_a + squares_b.T - 2 * a.dot(b.T))

        elif metric == "cosine":
            return 1 - self.xp.dot(
                a / self.xp.linalg.norm(a, axis=1)[:, None],
                (b / self.xp.linalg.norm(b, axis=1)[:, None]).T,
            )

        elif metric == "manhattan":
            func = lambda x, y: self.xp.sum(self.xp.abs(x.T - y), axis=-1)
            return self.xp.stack([func(a[i], b) for i in range(a.shape[0])])

        logger.error(
            "Available metrics are: " + '"euclidean", "cosine" and "manhattan"'
        )
        sys.exit(1)
