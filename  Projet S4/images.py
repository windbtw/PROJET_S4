import numpy as np
import cv2
import math
from random import *


class Images_aléaoires_NB:
    def __init__(self, taille, symétrie, taux_symétrie, clumpyness):
        """
        :param taille: taille de l'image
        :param symétrie: symétrie de l'image (Axiale, Centrale ou None)
        :param taux_symétrie: pourcentage déterminant à quel point l'image générée est symétrique (float compris entre 0 et 1)
        :param clumpyness: clumpyness de l'image (float entre 0 et 1)"""

        self.taille = taille
        self.symétrie = symétrie
        self.taux_symétrie = taux_symétrie
        self.clumpyness = clumpyness
        self.image = np.zeros([taille, taille], dtype=np.uint8)

    def draw(self):
        # symétrie axiale
        if self.symétrie == "Axiale":
            for i in range(self.taille):
                for j in range(self.taille // 2):
                    couleur = randint(0, 256)
                    self.image[i][j] = couleur
                    nb_symétrie = random()
                    if nb_symétrie <= self.taux_symétrie:
                        self.image[i][self.taille - j - 1] = couleur
                    else:
                        self.image[i][self.taille - j - 1] = randint(0, 256)

        # symétrie centrale
        if self.symétrie == "Centrale":
            for i in range(self.taille):
                for j in range(self.taille // 2):
                    couleur = randint(0, 256)
                    self.image[i][j] = couleur
                    nb_symétrie = random()
                    if nb_symétrie <= self.taux_symétrie:
                        self.image[self.taille - i -
                                   1][self.taille - j - 1] = couleur
                    else:
                        self.image[self.taille - i -
                                   1][self.taille - j - 1] = randint(0, 256)

        # pas de symétrie
        if self.symétrie is None:
            for i in range(self.taille):
                for j in range(self.taille):
                    couleur = randint(0, 256)
                    self.image[i][j] = couleur

        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


"""cv2.imshow("Image", symétrique(1000))
cv2.waitKey(0)
cv2.destroyAllWindows()"""
