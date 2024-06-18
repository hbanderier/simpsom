from typing import Optional
from nptyping import NDArray
import numpy as np
from numba import njit, prange
import os
os.environ["NUMBA_NUM_THREADS"] = "15"

@njit
def rolling_mean(img: NDArray, winsize: int, strides: int = 1) -> NDArray:
    img = np.ascontiguousarray(img)

    shape = (img.shape[0] - winsize + 1, img.shape[1] - winsize + 1, winsize, winsize)
    img_strides = img.strides
    img_strides = (img_strides[0], img_strides[1], img_strides[0], img_strides[1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=img_strides)[::strides, ::strides]
    output_img_shape = patches.shape[:2]
    patches = np.ascontiguousarray(patches).reshape(-1, winsize, winsize)

    output_img = np.array([np.mean(roi) for roi in patches])
    output_img = output_img.reshape(output_img_shape)
    return output_img


@njit
def one_ssim(im1: NDArray, im2: NDArray, win_size: int, strides: int = 1, data_range: float = 1.0) -> float:
    K1, K2 = 0.01, 0.03
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    
    ux = rolling_mean(im1, win_size, strides)
    uy = rolling_mean(im2, win_size, strides)

    # compute (weighted) variances and covariances
    uxx = rolling_mean(im1 * im1, win_size, strides)
    uyy = rolling_mean(im2 * im2, win_size, strides)
    uxy = rolling_mean(im1 * im2, win_size, strides)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2 // strides

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    S = 1 - S[pad:-pad, pad:-pad].mean()
    S = 0 if S < 0 else S
    S = 1 if S > 1 else S
    return S


@njit(parallel=True)
def pairwise_ssim_half(X: NDArray, win_size: int = 7, strides: int = 1, data_range: float = 1.0) -> NDArray:
    output = np.zeros((len(X), len(X)), dtype=np.float32)        
    for i in prange(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            im1, im2 = X[i], X[j]
            S = one_ssim(im1, im2, win_size, strides, data_range)
            output[i, j] = S
            output[j, i] = output[i, j]
    return output


@njit(parallel=True)
def pairwise_ssim_nothalf(X: NDArray, Y: NDArray, win_size: int = 7, strides: int = 1, data_range: float = 1.0) -> NDArray:
    output = np.zeros((len(X), len(Y)), dtype=np.float32)
    for i in prange(X.shape[0] - 1):
        for j in range(Y.shape[0]):
            im1, im2 = X[i], Y[j]
            S = one_ssim(im1, im2, win_size, strides, data_range)
            output[i, j] = S
    return output


def pairwise_ssim(X: NDArray, Y: Optional[NDArray] = None, win_size: int = 7, strides: int = 1, data_range: float = 1.0) -> NDArray:
    if Y is None:
        return pairwise_ssim_half(X, win_size, strides, data_range)
    return pairwise_ssim_nothalf(X, Y, win_size, strides, data_range)