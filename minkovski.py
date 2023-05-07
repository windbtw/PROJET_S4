import numpy as np
import matplotlib.pyplot as plt
from quantimpy import minkowski as mk
import cv2
from galaxy_functions import *

##function to get the Minkovski fonctionnal

def Minkowski_all(nom_image_entrée, nom_image_sortie):
    """
    plots the area,length and euler constant as a function of the threshold
    np.flip(dist) is a np matrice with the different values of the threshold
    area is a np matrice with the different areas of the pixels that are greater than the value of the threshold
    """
    img = cv2.imread(nom_image_entrée)
    gray_image = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # takes image and return a greyscales image because this fonction need a 2D image.
    # cv2.imwrite("greyscale.png", gray_image) # show the grey-scale image
    dist, area, length, euler = mk.functions_close(gray_image)

    threshold = np.flip(dist)
    plt.plot(threshold, area / 20, label="area/20")
    plt.plot(threshold, length, label="lenght")
    plt.plot(threshold, euler, label="euler")
    plt.legend()
    plt.xlabel(" threshold")
    plt.ylabel("Minkowski fonctional")
    plt.savefig(nom_image_sortie)


def Minkowski_fixed(nom_image_entrée, threshold):
    """
    aire, lenght and euler are based on all the ​​pixels whose values is greater than that of the threshold

    """
    img = load_image(nom_image_entrée)
    image_bool = img >= threshold
    aire, lenght, euler = mk.functionals(image_bool)
    return aire, lenght, euler


# (Minkowski_all("andromede.jpeg", "test_Minkowski"))
# print(Minkowski_fixed("andromede.jpeg", 200))