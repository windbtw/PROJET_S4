from code_for_database import *
from galaxie_generation_class import *
from galaxie_generation_main import *
from galaxy_functions import *
from get_data_function import *

##this is where we are executing the code for database generation, and image generation too.
#all folders in filepath have to be created before executing function


draw_many_images() ##you have to manually modify how many of each type you want inside the function, right it is set to low number for demo

filepath = "Images\\Galaxie_"    ##path where the pictures are (jpeg, if png modify inside code_for_database)                 
database_name = "dat files\Base_demo.dat"  ##path for database

get_data_informatique(filepath, database_name)

pritn("done")