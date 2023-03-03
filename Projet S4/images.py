import numpy as np
import cv2
import math
from random import *


class Images_aléaoires_NB:
    def __init__(self, taille, symétrie, taux_symétrie, taille_clumpyness, nb_pixel_clump):
        """
        :param taille: taille de l'image
        :param symétrie: symétrie de l'image (Axiale, Centrale ou None)
        :param taux_symétrie: pourcentage déterminant à quel point l'image générée est symétrique (float compris entre 0 et 1)
        :param taille_clumpyness: rayon des cercles ou cotés des carrés de la clumpyness
        :param nb_pixel_clump: nb de pixel à clumper sur l'image"""

        self.taille = taille
        self.symétrie = symétrie
        self.taux_symétrie = taux_symétrie
        self.taille_clumpyness = taille_clumpyness
        self.nb_pixel_clump = nb_pixel_clump
        self.image = np.zeros([taille, taille], dtype=np.uint8)

    def clump(self):

        for compteur in range(self.nb_pixel_clump):

            i = randint(self.taille_clumpyness,
                        self.taille - self.taille_clumpyness)

            j = randint(self.taille_clumpyness,
                        self.taille - self.taille_clumpyness)

            couleur1 = randint(0, 256)

            self.image[i][j] = couleur1

            # CERCLES
            for abscisse in range(i - self.taille_clumpyness, i + self.taille_clumpyness):
                for ordonnées in range(j - self.taille_clumpyness, j + self.taille_clumpyness):
                    if ((abscisse - i)**2 + (ordonnées - j)**2) <= self.taille_clumpyness ** 2:
                        self.image[abscisse][ordonnées] = couleur1
            # CARRES
            """for abscisse in range(i, i + taille_clumpyness):
                for ordonnées in range(j, j + taille_clumpyness):
                    image[abscisse][ordonnées] = couleur1"""

            compteur += 1

    def symétrie_fct(self):
        # symétrie axiale
        if self.symétrie == "Axiale":
            for i in range(self.taille):
                for j in range(self.taille // 2):
                    couleur = self.image[i][j]
                    nb_symétrie = random()
                    if nb_symétrie <= self.taux_symétrie:
                        self.image[i][self.taille - j - 1] = couleur
                    else:
                        self.image[i][self.taille - j - 1] = randint(0, 256)

        # symétrie centrale
        if self.symétrie == "Centrale":
            for i in range(self.taille):
                for j in range(self.taille // 2):
                    couleur = self.image[i][j]
                    nb_symétrie = random()
                    if nb_symétrie <= self.taux_symétrie:
                        self.image[self.taille - i -
                                   1][self.taille - j - 1] = couleur
                    else:
                        self.image[self.taille - i -
                                   1][self.taille - j - 1] = randint(0, 256)

    def draw(self):
        for i in range(self.taille):
            for j in range(self.taille):
                couleur = randint(0, 256)
                self.image[i][j] = couleur
        self.clump()
        self.symétrie_fct()

        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
