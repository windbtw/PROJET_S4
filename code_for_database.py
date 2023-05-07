from get_data_function import *
from function_fourier import *
from minkovski import *
from get_data_fits import *
import imageio.v2 as imageio
import csv
import os

##code for generate database, for fits and jpeg format, we have different function depending on witch type of images we are using, and what data we want as an output

def get_data_informatique(filepath, database_name):
    from csv import reader
    with open('liste_paramètres.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        # Passing the cav_reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)

    with open(database_name, "w") as f:     ##change the str here to choose the database
        f.write("C A S M G Area Lenght Euler TYPE1 Arms Name\n")                                 ##write the parameter of the returned by CAS_function.py
        for i in range(2):
            print(filepath + str(i) + ".jpeg")
            stat = get_galaxy_stat(filepath +"elliptique_" + str(i) + ".jpeg", None, False, image_format="jpeg")   ##if file type is fits dont forget to add the flux radius
            title = filepath +"elliptique_" + str(i) + ".jpeg"
            type_elliptique = list_of_rows[i][3]
            area, lenght, euler = Minkowski_fixed(title, 100)
            C, A, S, M, G = stat[0], stat[1], stat[2], stat[3], stat[4]
            C = C.astype('str')
            A = A.astype('str')
            S = S.astype('str')
            M = M.astype('str')
            G = G.astype('str')
            area = str(area)
            lenght = str(lenght)
            euler = str(euler)
            CAS = C + " "+ A + " "+ S+ " "+ M+ " "+ G+ " "+ area + " " + lenght + " " + euler + " " + str(0)+" "+str(0)+" "+ type_elliptique+ " "+"Galaxie_elliptique_" + str(i)
            f.write(CAS)
            f.write("\n")

        for n in range(2):
            for d in range(2):
                for a in range(2, 5):
                    print(filepath +"spirale_d" + str(d) + "a" + str(a) + "n" + str(n) + ".jpeg")
                    stat = get_galaxy_stat(filepath +"spirale_d" + str(d) + "a" + str(a) + "n" + str(n) + ".jpeg", None, False, image_format="jpeg")   ##if file type is fits dont forget to add the flux radius
                    title = filepath +"spirale_d" + str(d) + "a" + str(a) + "n" + str(n) + ".jpeg"
                    area, lenght, euler = Minkowski_fixed(title, 100)
                    C, A, S, M, G = stat[0], stat[1], stat[2], stat[3], stat[4]
                    C = C.astype('str')
                    A = A.astype('str')
                    S = S.astype('str')
                    M = M.astype('str')
                    G = G.astype('str')
                    area = str(area)
                    lenght = str(lenght)
                    euler = str(euler)
                    CAS = C + " "+ A + " "+ S+ " "+ M+ " "+ G+ " "+ area + " " + lenght + " " + euler + " " + str(1)+" "+ str(a)+" "+ 'NoneTypeSpiral' +" "+"Galaxie_spirale_d" + str(d) + "a" + str(a) + "n" + str(n)
                    f.write(CAS)
                    f.write("\n")

        f.close()


def get_data_fits(filepath, database_name):
    list_of_names = os.listdir("C:\\Users\\pilot\\Desktop\\projet_V1\\Images\\fits")
    with open(database_name, "w") as f:     ##change the str here to choose the database
        f.write("C A S M G Name\n")
        for name in list_of_names:
            if "fits" in name:                                 ##write the parameter of the returned by CAS_function.py
                print(name)
                stat = get_galaxy_stat(filepath + name , None, True, image_format="fits")   ##if file type is fits dont forget to add the flux radius
                C, A, S, M, G = stat[0], stat[1], stat[2], stat[3], stat[4]
                C = C.astype('str')
                A = A.astype('str')
                S = S.astype('str')
                M = M.astype('str')
                G = G.astype('str')
                CAS = C + " " + A + " " + S + " " + M + " " + G + " " + " " + name
                f.write(CAS)
                f.write("\n")
        f.close()


def get_data_fourier(filepath, database_name):

    with open(database_name, "w") as f:     ##change the str here to choose the database
        f.write("Name Fouriernfois\n")                                 ##write the parameter of the returned by CAS_function.py
        for i in range(2000):
            print(filepath + str(i) + ".jpeg")
            galaxie = imageio.imread(filepath +"elliptique_" + str(i) + ".jpeg")
            output = data_reduction(galaxie, 20)
            chaine = "Galaxie_elliptique_" + str(i) + " " + output
            f.write(chaine)
            f.write("\n")
        for n in range(330):
            for d in range(2):
                for a in range(2, 5):
                    galaxie = imageio.imread(filepath +"spirale_d" + str(d) + "a" + str(a) + "n" + str(n) + ".jpeg")
                    output = data_reduction(galaxie, 20)
                    chaine = "Galaxie_spirale_d" + str(d) + "a" + str(a) + "n" + str(n) + " " + output
                    f.write(chaine)
                    f.write("\n")
                    
    f.close()