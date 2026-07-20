#!/usr/bin/env python3

import numpy as np
"""
Performing a convolution on images using multiple kernels.
"""

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray of shape (m, h, w, c)
        kernels: numpy.ndarray of shape (kh, kw, c, nc)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels.")

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding

    elif padding == 'same':
        ph = int(np.ceil((((h - 1) * sh + kh - h) / 2)))
        pw = int(np.ceil((((w - 1) * sw + kw - w) / 2)))

    elif padding == 'valid':
        ph = 0
        pw = 0

    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w, nc))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            image_slice = padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]

            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    image_slice * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
