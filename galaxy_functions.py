import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.modeling.models import Sersic2D
from astropy.visualization import astropy_mpl_style
from scipy.ndimage import label

##some necessary functions for the conversion in black and white, to load an image etc

def load_image(image_path):
    # Load image with PIL
    with Image.open(image_path) as img:
        # Convert to grayscale numpy array
        img = img.convert('L')
        image_array = np.array(img)

    return image_array


def zoom(image, flux_radius):  # image must be a numpy array
    R1 = round((5 * flux_radius)/2)
    R2 = round((5 * flux_radius)/2)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    new_image = np.zeros((R1*2+1, R2*2+1))
    for i in range(center[0]-R1, center[0]+R1):
        for j in range(center[1]-R2, center[1]+R2):
            new_image[i-(center[0]-R1)][j-(center[1]-R2)] = image[i][j]
    return new_image


def convert_to_image(matrix):
    # invert the matrix so that black pixels become white and vice versa
    inverted_matrix = np.invert(matrix)
    # convert the numpy array to uint8 and scale to 0-255
    img_data = np.uint8(inverted_matrix * 255)
    # create a PIL image from the array
    img = Image.fromarray(img_data, 'L')
    # return the image
    return img


def plot_galaxy(image):
    plt.style.use(astropy_mpl_style)
    plt.imshow(image, origin='lower', cmap='gray')
    plt.colorbar()
    plt.show()












