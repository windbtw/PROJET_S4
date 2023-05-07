import numpy as np
import csv
import cv2
from random import *
from galaxie_generation_class import Galaxy

liste_paramètres = 'liste_paramètres.csv'

##all the drawing function

def draw_one_image(number, type, size, distribution, arms_number):

    if type == 1:
        image, parameters = Galaxy(type, size, distribution, arms_number).create_image()

        cv2.imwrite("Images\Galaxie_spirale_d" + str(distribution)+ "a" + str(arms_number) + "n" + str(number) + ".jpeg", image) # draws enter saves the image

        with open(liste_paramètres, mode='a', newline='') as fichier_csv:
            writer = csv.writer(fichier_csv) 
            writer.writerow(parameters) # Saves the parameters of each arms

    elif type == 0:
        image, parameters = Galaxy(type, size, distribution, arms_number).create_image()

        cv2.imwrite("Images\Galaxie_elliptique_" + str(number) + ".jpeg", image)

        with open(liste_paramètres, mode='a', newline='') as fichier_csv:
            writer = csv.writer(fichier_csv)
            writer.writerow(parameters)


def draw_images(number, type, size, distribution, arms_number):
    for i in range(number):
        print(i)
        draw_one_image(i, type, size, distribution, arms_number)


def draw_many_images():
    types= [0, 1]
    distributions = [0, 1]
    arms_numbers = [2, 3, 4]
    for type in types:
        if type == 1:
            for distribution in distributions:
                for arms_number in arms_numbers:
                    draw_images(2, type, 512, distribution, arms_number)
        if type ==0:
            draw_images(2, type, 512, distributions[0], arms_numbers[0])



