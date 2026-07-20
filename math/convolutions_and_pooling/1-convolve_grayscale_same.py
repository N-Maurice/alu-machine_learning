#!/usr/bin/env python3
'''
Performs a SAME convolution on grayscale images.
'''

import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w)
        kernel: numpy.ndarray of shape (kh, kw)

    Returns:
        numpy.ndarray containing the convolved images.
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    ph = kh // 2
    pw = kw // 2

    # Adjust padding for even-sized kernels
    if kh % 2 == 0:
        pad_top = ph
        pad_bottom = ph - 1
    else:
        pad_top = pad_bottom = ph

    if kw % 2 == 0:
        pad_left = pw
        pad_right = pw - 1
    else:
        pad_left = pad_right = pw

    # Pad the images with zeros
    padded = np.pad(
        images,
        (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right)
        ),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output
