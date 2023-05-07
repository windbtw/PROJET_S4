import cv2
import numpy as np
import math
from random import *
from scipy.optimize import fsolve

#class Galaxy, used to generate galaxies

class Galaxy:
    def __init__(self, type, image_size, spiral_distribution, arms_number):
        self.size = image_size
        self.image = np.zeros((image_size, image_size), dtype=np.uint8)
        self.type = type
        self.spiral_distribution = spiral_distribution
        self.arms_number = arms_number
        # Experimental value that gives an acceptable radius
        self.radius_spiral = int(np.round(image_size * 0.078125))

        """
        param image_size: pixel size of the image
        param self.image: initial numpy array contianing the image's data
        param type: type of the galaxy to generate, can be elliptic ( = 0) or spiral (= 1)
        param spiral_sitribution: distribution law of the spiral's clusters of points, can be gaussian or exponential (0 or 1)
        param self.radius_spiral: radius of the spiral's cluster of points
        param arms_number : number of arms of the spiral galaxy
        """

    def equation(self, x):  # fsolver equation
        a, b = x
        return [abs(a * (b ** (2 * math.pi))) + self.radius_spiral - self.size//2, 0]

    def create_image(self):
        if self.type == 0:

            a = randint(20, 150)/1000
            b = randint(150, 300)/1000
            orientation = randint(0, 360)
            sigma = 0.1

            n = round(10*(1-(a/b)))

            param_list = []
            param_list.append([a, b, orientation, f"{n}"])

            # Conversion of the rotation angle to radians
            theta_rad = math.radians(orientation)

            # Construction of the coordinate grid
            x, y = np.meshgrid(np.linspace(0, 1, self.size),
                               np.linspace(0, 1, self.size))

            # Rotation of the grid
            x_rot = (x-0.5) * np.cos(theta_rad) - \
                (y-0.5) * np.sin(theta_rad) + 0.5
            y_rot = (x-0.5) * np.sin(theta_rad) + \
                (y-0.5) * np.cos(theta_rad) + 0.5

            # Calculation of the distances of the points to the ellipse
            distances = np.sqrt(((x_rot - 0.5) / a) ** 2 +
                                ((y_rot - 0.5) / b) ** 2)

            # Mark the pixels inside the ellipse
            inside_ellipse = distances < 1

            # Create a copy of the image and set the intensity of the pixels inside the ellipse to a desired value
            intensities = np.exp(-0.5 * ((distances - 1) / sigma) ** 2) * 255
            intensities[inside_ellipse] = 255

            # Conversion of intensities to pixel intensity values
            intensities = intensities.astype(np.uint8)

            # Add intensities to the image's pixel array
            self.image[:, :] = intensities


        if self.type == 1:  # Constructing a spiral type galaxy with two spiral

            if self.arms_number == 2:

                #  Initializing the parameters of the spiral
                #  The spiral's equation is in polar coordinates r = a*(b**theta)

                #  Goal is to determine two parameters a and b, to generate a
                #  spiral that fits in the image size.
                #  To achieve this we use the relation abs(a*(b**(2*pi))) + radius < size//2,
                #  and the scipy.optimize library
                #  We generate roughly similar parameters to be able to get symetry properties

                angles = np.linspace(0, 1.9 * math.pi)  # angles to browse
                param_list = []  # List containing each arm's parameters

                #  Creating the first arm's parameters
                a_init_1 = uniform(-(self.size//15), self.size //
                                   15)  # Initial values of a
                b_init_1 = uniform(1.1            , 1.5)  # Initial values of b
                p_1 = a_init_1, b_init_1
                a, b = fsolve(self.equation, p_1)  # Solving the equation
                param_list.append([a, b])

                #  Creating the second arm's parameters
                a_init_2 = uniform(-a_init_1 - 5, -a_init_1 + 5)
                b_init_2 = uniform(b_init_1 - 0.05, b_init_1+ 0.05)
                p_2 = a_init_2, b_init_2
                a, b = fsolve(self.equation, p_2)
                param_list.append([a, b])

                # Manual parameters
                """a1 = 72
                a2 = -a1
                b = 1.31
                # list containing the parameters of each of the galaxy's arms
                param_list = [[a1, b], [a2, b]]"""

                #  Construction of the galaxy

                points_list = []  # List of points that verify the spiral's arms' parametric equation
                for param in param_list:
                    for theta in angles:
                        points_list.append(
                            [((param[0] * (param[1] ** theta)) * np.cos(theta)), ((param[0] * (param[1] ** theta)) * np.sin(theta))])

                for point in points_list:
                    point[0], point[1] = int(
                        np.round(point[0])), int(np.round(point[1]))  # Converting the values to integers

                for point in points_list: # Converting the points to fit the grid
                    point[0], point[1] = point[0] + \
                        self.size // 2, point[1] + self.size//2 

                for point in points_list:  # Creating the culsters of point of the spiral's axis
                    distribution = self.spiral_distribution
                    h = random()
                    radius = self.radius_spiral
                    mu = 20 * np.sqrt((self.size / 1024))  # Experimental value
                    if h <= 0.9:  # martin : changer valeur pour avoir des bras plus ou moins continus

                        # Creating two arms, that are roughly symetric, emerging from x = 0

                        if distribution == 0:
                            for x in range(point[0] - radius, point[0] + radius + 1):
                                for y in range(point[1] - radius, point[1] + radius + 1):
                                    if ((x - point[0])**2 + (y - point[1])**2) <= radius**2:
                                        intensity = np.round(
                                            255 * (np.exp(- (((x - point[0])**2) + ((y - point[1])**2))/((2*mu)**2))))  # Gaussian formula
                                        if intensity <= self.image[y][x]:
                                            continue
                                        else:
                                            self.image[y][x] = intensity

                        if distribution == 1:
                            # Experimental value
                            sigma = 0.035 * (1024 / self.size)
                            for x in range(point[0] - radius, point[0] + radius + 1):
                                for y in range(point[1] - radius, point[1] + radius + 1):
                                    if ((x - point[0])**2 + (y - point[1])**2) <= radius**2:
                                        intensity = np.round(
                                            255 * np.exp(-sigma * (np.sqrt((x - point[0])**2 + (y - point[1])**2))))
                                        if intensity <= self.image[y][x]:
                                            continue
                                        else:
                                            self.image[y][x] = intensity

            if self.arms_number == 3:

                angles = np.linspace(0, 1.9 * math.pi)  # angles to browse
                param_list = []  # List containing each arm's parameters

                # To create three arms, we use a similar method to the previous code
                # We generate parametersq for the 3 arms, but we're going to rotate our matrix
                # Once an arm in generated, we print it on the matrix, then rotate it by 120 degrees

                #  Creating the first arm's parameters
                a_init_1 = uniform(0, self.size // 15)  # Initial values of a
                b_init_1 = uniform(1.1, 1.5)  # Initial values of b
                p_1 = a_init_1, b_init_1
                a, b = fsolve(self.equation, p_1)  # Solving the equation
                param_list.append([a, b])

                #  Creating the second arm's parameters
                a_init_2 = uniform(a_init_1 - 5, a_init_1 + 5)
                b_init_2 = uniform(b_init_1 - 0.005, b_init_1 + 0.005)
                p_2 = a_init_2, b_init_2
                a, b = fsolve(self.equation, p_2)
                param_list.append([a, b])
                
                #  Creating the thirs arm's parameters

                a_init_3 = uniform(a_init_1 - 5, a_init_1 + 5)
                b_init_3 = uniform(b_init_1 - 0.005, b_init_1 + 0.005)
                p_3 = a_init_3, b_init_3
                a, b = fsolve(self.equation, p_3)
                param_list.append([a, b])

                points_list = [] # List of point that verify the spiral's equation
                alpha = 0  # Rotation angle of the arms

                for param in param_list: 
                    for theta in angles:
                        x_i, y_i = (param[0] * (param[1] ** theta)) * np.cos(theta), (param[0] * (param[1] ** theta)) * np.sin(theta) # initial points.
                        x_f = (x_i * np.cos(alpha)) - (y_i * np.sin(alpha)) # rotation of x
                        y_f = (y_i * np.cos(alpha)) + (x_i * np.sin(alpha)) # rotation of y

                        
                        points_list.append([x_f, y_f])

                    alpha += (2*np.pi)/3
                
                for point in points_list:
                    point[0], point[1] = int(
                        np.round(point[0])), int(np.round(point[1]))  # Converting the values to integers

                for point in points_list: # Converting the points to fit the grid
                    point[0], point[1] = point[0] + \
                        self.size // 2, point[1] + self.size//2
                
                for point in points_list:  # Creating the culsters of point of the spiral's axis
                    distribution = self.spiral_distribution
                    h = random()
                    radius = self.radius_spiral
                    mu = 15 * np.sqrt((self.size / 1024))  # Experimental value
                    if h <= 0.8:  
                        if distribution == 0:
                            for x in range(point[0] - radius, point[0] + radius + 1):
                                for y in range(point[1] - radius, point[1] + radius + 1):
                                    if ((x - point[0])**2 + (y - point[1])**2) <= radius**2:
                                        intensity = np.round(
                                            255 * (np.exp(- (((x - point[0])**2) + ((y - point[1])**2))/((2*mu)**2))))  # Gaussian formula
                                        if intensity <= self.image[y][x]:
                                            continue
                                        else:
                                            self.image[y][x] = intensity

                        if distribution == 1:
                            # Experimental value
                            sigma = 0.032 * (1024 / self.size)
                            for x in range(point[0] - radius, point[0] + radius + 1):
                                for y in range(point[1] - radius, point[1] + radius + 1):
                                    if ((x - point[0])**2 + (y - point[1])**2) <= radius**2:
                                        intensity = np.round(
                                            255 * np.exp(-sigma * (np.sqrt((x - point[0])**2 + (y - point[1])**2))))
                                        if intensity <= self.image[y][x]:
                                            continue
                                        else:
                                            self.image[y][x] = intensity
            
            if self.arms_number == 4:
                angles = np.linspace(0, 1.9 * math.pi)  # angles to browse
                param_list = []  # List containing each arm's parameters

                # To create four arms, we use a similar method to the previous code
                # We generate parametersq for the 4 arms, but we're going to rotate our matrix
                # Once an arm in generated, we print it on the matrix, then rotate it by 90 degrees

                #  Creating the first arm's parameters
                a_init_1 = uniform(0, self.size // 10)  # Initial values of a
                b_init_1 = uniform(1.1, 1.5)  # Initial values of b
                p_1 = a_init_1, b_init_1
                a, b = fsolve(self.equation, p_1)  # Solving the equation
                param_list.append([a, b])

                #  Creating the second arm's parameters
                a_init_2 = uniform(a_init_1 - 5, a_init_1 + 5)
                b_init_2 = uniform(b_init_1 - 0.005, b_init_1 + 0.005)
                p_2 = a_init_2, b_init_2
                a, b = fsolve(self.equation, p_2)
                param_list.append([a, b])
                
                #  Creating the thirs arm's parameters

                a_init_3 = uniform(a_init_1 - 5, a_init_1 + 5)
                b_init_3 = uniform(b_init_1 - 0.005, b_init_1 + 0.005)
                p_3 = a_init_3, b_init_3
                a, b = fsolve(self.equation, p_3)
                param_list.append([a, b])

                #  Creating the fourth arm's parameters

                a_init_4 = uniform(a_init_1 - 5, a_init_1 + 5)
                b_init_4 = uniform(b_init_1 - 0.005, b_init_1 + 0.005)
                p_4 = a_init_4, b_init_4
                a, b = fsolve(self.equation, p_4)
                param_list.append([a, b])



                points_list = [] # List of point that verify the spiral's equation
                alpha = 0  # Rotation angle of the arms

                for param in param_list: 
                    for theta in angles:
                        x_i, y_i = (param[0] * (param[1] ** theta)) * np.cos(theta), (param[0] * (param[1] ** theta)) * np.sin(theta) # initial points.
                        x_f = (x_i * np.cos(alpha)) - (y_i * np.sin(alpha)) # rotation of x
                        y_f = (y_i * np.cos(alpha)) + (x_i * np.sin(alpha)) # rotation of y

                        
                        points_list.append([x_f, y_f])

                    alpha += np.pi / 2
                
                for point in points_list:
                    point[0], point[1] = int(
                        np.round(point[0])), int(np.round(point[1]))  # Converting the values to integers

                for point in points_list: # Converting the points to fit the grid
                    point[0], point[1] = point[0] + \
                        self.size // 2, point[1] + self.size//2
                
                for point in points_list:  # Creating the culsters of point of the spiral's axis
                    distribution = self.spiral_distribution
                    h = random()
                    radius = self.radius_spiral
                    mu = 15 * np.sqrt((self.size / 1024))  # Experimental value
                    if h <= 0.9:  
                        if distribution == 0:
                            for x in range(point[0] - radius, point[0] + radius + 1):
                                for y in range(point[1] - radius, point[1] + radius + 1):
                                    if ((x - point[0])**2 + (y - point[1])**2) <= radius**2:
                                        intensity = np.round(
                                            255 * (np.exp(- (((x - point[0])**2) + ((y - point[1])**2))/((2*mu)**2))))  # Gaussian formula
                                        if intensity <= self.image[y][x]:
                                            continue
                                        else:
                                            self.image[y][x] = intensity

                        if distribution == 1:
                            # Experimental value
                            sigma = 0.035 * (1024 / self.size)
                            for x in range(point[0] - radius, point[0] + radius + 1):
                                for y in range(point[1] - radius, point[1] + radius + 1):
                                    if ((x - point[0])**2 + (y - point[1])**2) <= radius**2:
                                        intensity = np.round(
                                            255 * np.exp(-sigma * (np.sqrt((x - point[0])**2 + (y - point[1])**2))))
                                        if intensity <= self.image[y][x]:
                                            continue
                                        else:
                                            self.image[y][x] = intensity
        
        # Adding random clusters of point that simutate stars

        h = randint(0, 8)
        stars_radius = self.radius_spiral // 2
        for i in range(h):
            x_0 = randint(stars_radius + 1 , self.size - stars_radius - 1) # Centers of the star
            y_0 = randint(stars_radius + 1 , self.size - stars_radius - 1)
            sigma = 0.035 * (1024 / self.size)
            for x in range(x_0 - stars_radius, x_0 + stars_radius + 1):
                for y in range(y_0 - stars_radius, y_0 + stars_radius + 1):
                    if ((x - x_0)**2 + (y - y_0)**2) <= stars_radius**2:
                        intensity = np.round(
                             255 * np.exp(-sigma * (np.sqrt((x - x_0)**2 + (y - y_0)**2))))
                        if intensity <= self.image[y][x]:
                             continue
                        else:
                            self.image[y][x] = intensity



        # Adding the noise

        for x in range(self.size):
            for y in range(self.size):
                g = random()
                if g <= 0.5:
                    self.image[y][x] = np.random.normal(
                        125, 20)  # gaussian noise
        return self.image, param_list[0]

        """# Display of the image
        cv2.imshow("Image de galaxie", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
