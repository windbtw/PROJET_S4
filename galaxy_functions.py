import numpy as np
from PIL import Image
from scipy.ndimage import label


def load_image(image_path):
    # Load image with PIL
    with Image.open(image_path) as img:
        # Convert to grayscale numpy array
        img = img.convert('L')
        image_array = np.array(img)

    return image_array


def concentration_index(image):
    # Determine the center of the image
    y, x = np.indices(image.shape)
    xbar = (x * image).sum() / image.sum()
    ybar = (y * image).sum() / image.sum()

    # Compute the distance from each pixel to the center
    r = np.sqrt((x - xbar) ** 2 + (y - ybar) ** 2)

    # Compute the cumulative flux within each radius
    cum_flux = np.zeros(len(r))
    for i in range(len(r)):
        cum_flux[i] = image[r <= r[i]].sum()

    # Compute the concentration index
    C = 5 * np.log10(cum_flux[-1] / cum_flux[0])
    return C


def compute_asymmetry(image):
    # flip the image horizontally and subtract it from the original image
    flipped = np.fliplr(image)
    diff = image - flipped

    # normalize the difference image
    norm = np.sum(image)
    norm_diff = diff / norm

    # compute the asymmetry index
    asym = np.sum(np.abs(norm_diff))

    return asym











