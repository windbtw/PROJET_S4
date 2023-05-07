import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import *
from typing import Callable, Tuple

##that part is only to get the furier function

def matriceImage(matrice: np.ndarray, gamma: float, rgb: tuple) -> np.ndarray:
    """
    Converts a matrix of values in a color space to an RGB image.

    Args:
        matrice: A matrix of values in a color space.
        gamma: The gamma value to use for exponentiation.
        rgb: A tuple containing the RGB conversion coefficients.

    Returns:
        An RGB image representing the input matrix.

    """
    # Get the shape of the input matrix
    shape = matrice.shape

    # Calculate the exponent value a
    a = 1.0 / gamma

    # Normalize the matrix by the maximum value
    norm = matrice.max()
    m = np.power(matrice / norm, a)

    # Create an empty image matrix
    im = np.zeros((shape[0], shape[1], 3), dtype=np.float64)

    # Apply the RGB coefficients to the normalized matrix
    im[:, :, 0] = np.real(rgb[0] * m)
    im[:, :, 1] = np.real(rgb[1] * m)
    im[:, :, 2] = np.real(rgb[2] * m)

    return im


def matriceImageLog(matrice: np.ndarray, rgb: tuple) -> np.ndarray:
    """
    Converts a matrix of values in a color space to an RGB image, with a logarithmic transformation.

    Args:
        matrice: A matrix of values in a color space.
        rgb: A tuple containing the RGB conversion coefficients.

    Returns:
        An RGB image representing the input matrix.

    """
    # Get the shape of the input matrix
    shape = matrice.shape

    # Apply the logarithmic transformation to the input matrix
    m = np.log10(1 + matrice)

    # Normalize the transformed matrix
    m_min = m.min()
    m_max = m.max()
    m_norm = (m - m_min) / (m_max - m_min)

    # Create an empty image matrix
    im = np.zeros((shape[0], shape[1], 3), dtype=np.float64)

    # Apply the RGB coefficients to the normalized matrix
    im[:, :, 0] = np.real(rgb[0] * m_norm)
    im[:, :, 1] = np.real(rgb[1] * m_norm)
    im[:, :, 2] = np.real(rgb[2] * m_norm)

    return im


def plotSpectre(image: np.ndarray, Lx: float, Ly: float, legend: str) -> None:
    """
    Plots the frequency spectrum of an image.

    Args:
        image: An RGB image.
        Lx: The length of the image in the x direction.
        Ly: The length of the image in the y direction.

    Returns:
        None.

    """
    # Get the dimensions of the input image
    (Ny, Nx) = image.shape[0:2]

    # Calculate the maximum frequency values
    fxm = Nx * 1.0 / (2 * Lx)
    fym = Ny * 1.0 / (2 * Ly)

    # Display the image with the frequency extent
    imshow(image, extent=[-fxm, fxm, -fym, fym])
    xlabel("fx")
    ylabel("fy")
    plt.savefig(legend)
    plt.show()


def transfert_passe_bas(fx: float, fy: float, p: list) -> float:
    """
    Computes the transfer function of a low-pass filter.

    Args:
        fx: The frequency in the x direction.
        fy: The frequency in the y direction.
        p: A list of filter parameters, where the first element is the cutoff frequency.

    Returns:
        The transfer function value.

    """
    # Extract the cutoff frequency from the filter parameters
    fc = p[0]

    # Calculate the transfer function
    return 1.0 / np.power(1 + (fx * fx + fy * fy) / (fc * fc), 2)


def matriceFiltre(
    matrice: np.ndarray,
    transfert: Callable[[float, float, Tuple[float]], float],
    p: Tuple[float],
) -> np.ndarray:
    """
    Applies a given transfer function to a matrix and returns the resulting matrix.

    Args:
    - matrice (np.ndarray): the input matrix
    - transfert (Callable[[float, float, Tuple[float]], float]): a transfer function
    - p (Tuple[float]): a tuple of parameters needed by the transfer function

    Returns:
    - np.ndarray: the resulting matrix
    """
    s = matrice.shape
    Nx = s[1]
    Ny = s[0]
    nx = Nx / 2
    ny = Ny / 2
    Mat = np.zeros((Ny, Nx), dtype=np.complex128)
    for n in range(Nx):
        for k in range(Ny):
            fx = float(n - nx - 1) / Nx
            fy = float(k - ny - 1) / Ny
            Mat[k, n] = matrice[k, n] * transfert(fx, fy, p)
    return Mat


def TFD(image: np.ndarray) -> np.ndarray:
    """
    Applies the 2D Fourier Transform on a grayscale image and returns its power spectrum.

    Args:
        image (np.ndarray): the grayscale input image

    Returns:
        np.ndarray: the power spectrum of the Fourier transform
    """
    # Convert the image to a float64 numpy array
    image_array = np.array(image[:, :], dtype=np.float64)

    # Compute the 2D discrete Fourier transform
    discrete_fourier = fft2(image_array)

    # Center the discrete Fourier transform
    centered_discrete_fourier = fftshift(discrete_fourier)

    return centered_discrete_fourier


def square_module(ftc):
    P = np.power(np.absolute(ftc), 2)
    return P


def module_matrice(matrice):
    """
    Computes the module of a complex-valued numpy matrix

    Arguments:
    matrice -- a complex-valued numpy matrix

    Returns:
    a numpy array containing the modules of each element of the matrix
    """
    # Compute the absolute value of each element of the matrix using the numpy function `np.abs`
    # This corresponds to the modulus for complex numbers
    return np.abs(matrice)


def porte_nul(shape):
    """
    This function returns a numpy array filled with zeros.

    Args:
    - shape: int, the size of the array

    Returns:
    - np.array, an array filled with zeros of shape (shape, shape)
    """
    return np.zeros((shape, shape))


def porte_courone(shape, k, dk, verbose=False):
    """
    This function returns a numpy array representing a ring-shaped filter, also known as a 'couronne' filter.

    Args:
    - shape: int, the size of the array
    - k: int, the radius of the ring
    - dk: int, the width of the ring
    - verbose: bool, optional, whether to print messages or not

    Returns:
    - np.array, an array filled with zeros except for a ring-shaped filter with radius k and width dk, centered in the middle of the array
    """
    a = porte_nul(shape)
    centre = a.shape[0] // 2
    if dk == k:
        for x in range(0, a.shape[0]):
            for y in range(0, a.shape[1]):
                if (x - centre) ** 2 + (y - centre) ** 2 <= k**2:
                    a[y][x] = 1

    else:
        for x in range(0, a.shape[0]):
            for y in range(0, a.shape[1]):
                if (k - dk) ** 2 <= (x - centre) ** 2 + (y - centre) ** 2 <= (k + dk) ** 2:
                    a[y][x] = 1

    return a


def moyenne_courrone(array, k, dk):
    """
    This function calculates the average value of an array within a ring-shaped region.

    Args:
    - array: np.array, an input array
    - k: int, the radius of the ring
    - dk: int, the width of the ring

    Returns:
    - float, the average value of the array within the ring-shaped region with radius k and width dk, centered in the middle of the array
    """
    centre = array.shape[0] // 2
    somme = 0
    denominateur = 0
    for u in range(0, array.shape[1]):
        for v in range(0, array.shape[0]):
            if (k - dk) ** 2 <= (u - centre) ** 2 + (v - centre) ** 2 <= (k + dk) ** 2:
                somme += array[v][u]
                denominateur += 1
    if denominateur == 0:
        return 0
    else:
        return somme / denominateur


def data_reduction(array, dk):
    """
    This function performs data reduction on an input array by calculating the average value of a ring-shaped region of the Fourier transform.

    Args:
    - array: np.array, an input array
    - dk: int, the width of the ring

    Returns:
    - tuple of two np.arrays, containing the x and y values of the data reduction plot
    """
    shape = array.shape
    VC = TFD(array)  # Fourier transform
    module_TF = module_matrice(VC)  # modulus of the Fourier transform
    liste = []
    for k in range(dk, shape[0] // 2, dk):
        W = porte_courone(shape[0], k, dk)
        J = np.multiply(module_TF, W)
        liste.append(moyenne_courrone(J, k, dk))

    yhash = liste
    xhash = np.linspace(dk, shape[0] // 2, len(yhash))

    chaine = ""
    for k in yhash:
        chaine += str(k) + " "
    return chaine


def makeGaussian(size, fwhm=400, center=None):
    """Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)
