import numpy as np
from astropy.modeling.models import Sersic2D
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style


def generate_galaxy(amplitude, r_eff, n, x_0, y_0,ellip, theta):
    galaxy_model = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    y, x = np.mgrid[0:256, 0:256]
    image = galaxy_model(x, y)
    return image


def plot_galaxy(image):
    plt.style.use(astropy_mpl_style)
    plt.imshow(image, origin='lower', cmap='gray')
    plt.colorbar()
    plt.show()