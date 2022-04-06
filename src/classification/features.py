import numpy as np
from scipy.fftpack import dct as sc_dct
import cv2
import matplotlib.pyplot as plt


# Histogram feature
def histogram(image, BINS=30, SHOW=False):
    hist, bin_edges = np.histogram(image, bins=BINS, range=(0, 1))
    if SHOW:
        plt.imshow(image, cmap="gray"), plt.xticks([]), plt.yticks([])
        plt.title("Original")
        plt.show()
        plt.hist(image.flatten(), bins=BINS, range=(0, 1))
        plt.title(f"hist, BINS={BINS}")
        plt.show()
    return hist


# DFT feature
def dft(image, p=13, SHOW=False):
    f = np.fft.fft2(image)
    f = np.abs(f[0:p, 0:p])
    zigzag = []
    for index in range(1, p + 1):
        slice = [row[:index] for row in f[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    if SHOW:
        plt.imshow(f, cmap="gray")
        plt.title(f"dft, p={p}")
        plt.show()
    return zigzag[1:]


def show_dft(image, p=13):
    f = np.fft.fft2(image)
    return np.abs(f[0:p, 0:p])


# DCT feature
def dct(image, p=13, SHOW=False):
    c = sc_dct(image, axis=1)
    c = sc_dct(c, axis=0)
    c = c[0:p, 0:p]
    zigzag = []
    for index in range(1, p + 1):
        slice = [row[:index] for row in c[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    if SHOW:
        plt.imshow(c, cmap="gray")
        plt.title(f"dct, p={p}")
        plt.show()
    return zigzag


def show_dct(image, p=13):
    c = sc_dct(image, axis=1)
    c = sc_dct(c, axis=0)
    return c[0:p, 0:p]


# Gradient feature
def gradient(image, n=2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result


# Scale feature
def scale(image, scale=0.35, SHOW=False):
    height = image.shape[0]
    width = image.shape[1]
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size,
                               interpolation=cv2.INTER_AREA)
    if SHOW:
        plt.imshow(resized_image, cmap="gray")
        plt.title(f"scale, sc={scale}")
        plt.show()
    return resized_image
