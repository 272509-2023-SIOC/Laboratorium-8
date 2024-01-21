# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from skimage import color, data, io
from skimage.transform import resize
import pywt


# Funkcja do kompresji obrazu przy użyciu Transformacji Fouriera
def compress_image_fourier(image, keep_fraction):
    """
       Compress an image using the Fourier Transform.

    """
    # Konwersja do skali szarości jeśli obraz jest kolorowy
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Przeprowadzenie szybkiej transformacji Fouriera
    im_fft = fft2(image)

    # Ustalenie wymiarów obrazu
    rows, cols = image.shape

    # Usunięcie części współczynników
    rows_keep = int(rows * keep_fraction)
    cols_keep = int(cols * keep_fraction)
    im_fft[int(rows_keep / 2):-int(rows_keep / 2)] = 0
    im_fft[:, int(cols_keep / 2):-int(cols_keep / 2)] = 0

    # Odwrotna transformacja Fouriera
    new_image = ifft2(im_fft).real

    return new_image

def adaptive_compression_coefficient_fourier(image):
    """
    Calculate an adaptive compression coefficient for Fourier transform.
    """
    im_fft = fft2(image)
    # Tutaj możesz użyć różnych metod do obliczenia progu, np. na podstawie średniej wartości
    threshold = np.abs(im_fft).mean()
    return threshold
def compress_image_wavelet_v2(image, level, threshold):
    """
    Compress an image using Wavelet Transform with corrected implementation.
    """
    # Konwersja do skali szarości jeśli obraz jest kolorowy
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Przeprowadzenie transformacji falkowej
    coeffs = pywt.wavedec2(image, 'haar', level=level)

    # Zerowanie małych współczynników przy użyciu obliczonego progu
    coeffs = list(coeffs)
    for i in range(1, len(coeffs)):
        coeffs[i] = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs[i])

    # Odwrotna transformacja falkowa
    new_image = pywt.waverec2(coeffs, 'haar')

    # Obcięcie obrazu do oryginalnych wymiarów
    new_image = new_image[:image.shape[0], :image.shape[1]]

    return new_image
def adaptive_compression_coefficient_wavelet(image, level):
    """
    Calculate an adaptive compression coefficient for wavelet transform.
    """
    coeffs = pywt.wavedec2(image, 'haar', level=level)
    # Oblicz próg na podstawie wariancji lub innej statystyki
    threshold = np.max([np.max(np.abs(c)) for c in coeffs[1:]]) * 0.01
    return threshold


# Załaduj obraz
image = data.camera()  # Przykładowy obraz z biblioteki skimage

# Dla Transformacji Fouriera
fourier_threshold = adaptive_compression_coefficient_fourier(image)
compressed_image_fourier = compress_image_fourier(image, fourier_threshold)

# Dla Transformacji Falkowej
level = 2  # lub inna wartość, zależnie od potrzeb
wavelet_threshold = adaptive_compression_coefficient_wavelet(image, level)
compressed_image_wavelet_v2 = compress_image_wavelet_v2(image, level, wavelet_threshold)

# Wyświetlenie oryginalnego obrazu i obrazów po kompresji
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("Oryginalny obraz")
ax[1].imshow(compressed_image_fourier, cmap=plt.cm.gray)
ax[1].set_title("Po kompresji Fourierem")
ax[2].imshow(compressed_image_wavelet_v2, cmap=plt.cm.gray)
ax[2].set_title("Po kompresji Falkową (poprawione v2)")

plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
