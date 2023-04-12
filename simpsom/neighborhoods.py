from types import ModuleType
from typing import Union, Callable, Tuple, Any
from collections.abc import Sequence
from numpy.typing import ArrayLike
from functools import partial

import numpy as np

try:
    import cupy as cp
except ImportError:
    pass
from loguru import logger


class Neighborhoods:
    """Container class with functions to calculate neihgborhoods."""

    def __init__(
        self,
        xp: ModuleType,
        width: int,
        height: int,
        polygons: str,
        dist_type: str | Sequence[str] = "grid",
        PBC: bool = False,
    ) -> None:
        """Instantiate the Neighborhoods class.

        Args:
            xp (numpy or cupy): the numeric labrary to use
                to calculate distances.
        """
        self.xp = xp
        self.width = width
        self.height = height
        self.PBC = PBC
        self.polygons = polygons
        self.distances = self.compute_distances(dist_type)

    def compute_distances(
        self, dist_type: str | Sequence[str] = "grid"
    ) -> ArrayLike:
        if not isinstance(dist_type, str):
            self.sub_dist_type = dist_type[1]
            self.dist_type = dist_type[0]
        elif self.polygons.lower() == 'hexagons' and dist_type == 'grid':
            self.dist_type = dist_type
            self.sub_dist_type = "chebyshev"
        elif dist_type == 'grid':
            self.dist_type = dist_type
            self.sub_dist_type = 'l1'
        else:
            self.dist_type = dist_type
            self.sub_dist_type = 'l2'
        self.coordinates = self.xp.asarray(
            [[x, y] for x in range(self.width) for y in range(self.height)]
        ).astype(np.float32)
        if self.polygons.lower() == "hexagons":
            oddrows = self.coordinates[:, 1] % 2 == 1
            self.coordinates[:, 1] *= np.sqrt(3) / 2
            self.coordinates[oddrows, 0] += 0.5
        if dist_type == "cartesian":
            return self.cartesian_distances(
                self.coordinates, self.coordinates
            )
        input = self.xp.arange(self.height * self.width)
        return self.grid_distance(input, input)

    def cartesian_distances(
        self,
        a: ArrayLike,
        b: ArrayLike,
    ) -> ArrayLike:
        self.dx = self.xp.abs(b[:, None, 0] - a[None, :, 0])
        self.dy = self.xp.abs(b[:, None, 1] - a[None, :, 1])
        if self.PBC:
            maxdx = self.width
            if self.polygons.lower() == "hexagons":
                maxdy = self.height * np.sqrt(3) / 2
            else:
                maxdy = self.height
            maskx = self.dx > maxdx / 2
            masky = self.dy > maxdy / 2
            self.dx[maskx] = maxdx - self.dx[maskx]
            self.dy[masky] = maxdy - self.dy[masky]
        if self.sub_dist_type.lower() in ["l1", "manhattan"]:
            return self.dx + self.dy
        elif self.sub_dist_type.lower() in ["linf", "chebyshev"]:
            return self.xp.amax([self.dx, self.dy], axis=0)
        return self.xp.sqrt(self.dx ** 2 + self.dy ** 2)

    def hexagonal_grid_distance(
        self, xi: ArrayLike, yi: ArrayLike, xj: ArrayLike, yj: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        dy = yj[None, :] - yi[:, None]
        dx = xj[None, :] - xi[:, None]
        corr = xj[None, :] // 2 - xi[:, None] // 2
        if self.PBC:
            maskx = self.xp.abs(dx) > (self.width / 2)
            masky = self.xp.abs(dy) > (self.height / 2)
            dx[maskx] = -self.xp.sign(dx[maskx]) * (self.width - self.xp.abs(dx[maskx]))
            dy[masky] = -self.xp.sign(dy[masky]) * (
                self.height - self.xp.abs(dy[masky])
            )
            corr[maskx] = -self.xp.sign(corr[maskx]) * (
                self.width // 2 - self.xp.abs(corr[maskx])
            )
        dy = dy - corr
        return dx, dy

    def rectangular_grid_distance(
        self, xi: ArrayLike, yi: ArrayLike, xj: ArrayLike, yj: ArrayLike
    ) -> ArrayLike:
        dy = self.xp.abs(yj[None, :] - yi[:, None])
        dx = self.xp.abs(xj[None, :] - xi[:, None])
        if self.PBC:
            maskx = self.xp.abs(dx) > (self.width / 2)
            masky = self.xp.abs(dy) > (self.height / 2)
            dx[maskx] = self.width - dx[maskx]
            dy[masky] = self.height - dx[maskx]
        return dx, dy

    def grid_distance(
        self, i: ArrayLike, j: ArrayLike
    ) -> ArrayLike:
        ndim = 0
        i, j = self.xp.asarray(i), self.xp.asarray(j)
        nx, ny = self.width, self.height
        for input in [i, j]:
            ndim += input.ndim
        i, j = self.xp.atleast_1d(i), self.xp.atleast_1d(j)
        yi, xi = divmod(i, nx)
        yj, xj = divmod(j, ny)
        if self.polygons.lower() == "hexagons":
            self.dx, self.dy = self.hexagonal_grid_distance(xi, yi, xj, yj)
            mask = self.xp.sign(self.dx) == self.xp.sign(self.dy)
            all_dists = self.xp.where(
                mask,
                self.xp.abs(self.dx + self.dy),
                self.xp.amax([self.xp.abs(self.dx), self.xp.abs(self.dy)], axis=0),
            )
        else:
            self.dx, self.dy = self.rectangular_grid_distance(xi, yi, xj, yj)
            if self.sub_dist_type.lower() in ["linf", "chebyshev"]:
                all_dists = self.xp.amax([self.dx, self.dy], axis=0)
            else:
                all_dists = self.dx + self.dy
        if ndim == 0:
            return all_dists[0, 0]
        elif ndim == 1:
            return all_dists.flatten()
        return all_dists

    def gaussian(self, c: ArrayLike, denominator: float) -> ArrayLike:
        """Gaussian neighborhood function.

        Args:
            c (ArrayLike): center points.
            denominator (float): the 2sigma**2 value.

        Returns
            (ArrayLike): neighbourhood function between the points in c and all points.
        """
        if self.dist_type == 'cartesian':
            thetax = self.xp.exp(-self.xp.power(self.dx[c], 2) / denominator)
            thetay = self.xp.exp(-self.xp.power(self.dy[c], 2) / denominator)
            return thetax * thetay
        return self.xp.exp(-self.xp.power(self.distances[c], 2) / denominator)

    def mexican_hat(self, c: ArrayLike, denominator: float) -> ArrayLike:
        """Mexican hat neighborhood function.

        Args:
            c (ArrayLike): center points.
            denominator (float): the 2sigma**2 value.

        Returns
            (ArrayLike): neighbourhood function between the points in c and all points.
        """
        if self.dist_type == 'cartesian':
            thetax = self.xp.power(self.dx[c], 2)
            thetay = self.xp.power(self.dy[c], 2)
            theta = thetax + thetay
        else:
            theta = self.xp.power(self.distances[c], 2)
        return (1 - 2 * theta / denominator) * self.xp.exp(- theta / denominator)

    def bubble(self, c: ArrayLike, threshold: float) -> ArrayLike:
        """Bubble neighborhood function.

        Args:
            c (ArrayLike): center points.
            threshold (float): the bubble threshold.
        Returns
            (ArrayLike): neighbourhood function between the points in c and all points.
        """
        if self.dist_type == 'cartesian':
            return (self.dx[c] < threshold).astype(np.float32) * (self.dy[c] < threshold).astype(np.float32)
        return (self.distances[c] < threshold).astype(np.float32)

    def neighborhood_caller(
        self,
        centers: ArrayLike,
        sigma: float,
        neigh_func: str,
    ) -> ArrayLike:
        """Returns a neighborhood selection on any 2d topology.

        Args:
            center (Tuple[ArrayLike]): index of the center point along the xx yy grid.
            sigma (float): standard deviation/size coefficient.
            nigh_func (str): neighborhood specific distance function name
                (choose among 'gaussian', 'mexican_hat' or 'bubble')

        Returns:
            (array): the resulting neighborhood matrix.
        """

        d = 2 * sigma ** 2

        if neigh_func == "gaussian":
            return self.gaussian(centers, denominator=d)
        elif neigh_func == "mexican_hat":
            return self.mexican_hat(centers, denominator=d)
        elif neigh_func == "bubble":
            return self.bubble(centers, threshold=sigma)
        else:
            logger.error(
                "{} neighborhood function not recognized.".format(neigh_func)
                + "Choose among 'gaussian', 'mexican_hat' or 'bubble'."
            )
            raise ValueError
