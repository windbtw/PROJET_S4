import cv2
import numpy as np
import random
import math

class Matrix():
    #takes an image in the file and store its matrix in self.matrix
    #image needs to be : "image.png/jpg"

    def __init__(self, image):
        self.matrix = cv2.imread(image, 0)

    #takes self.matrix and convert it into an image stored in the file
    #title needs to be : "title.png/jpg"

    def export(self, title):
        cv2.imwrite(title, self.matrix)

    #returns a tuple with the dimension of self.matrix
    def get_dimension(self):
        x = np.size(self.matrix, 0)
        y = np.size(self.matrix, 1)
        return((x,y))

    #make the image brighter by adding the argument value to every pixel
    def brighter(self, value):
        (x, y) = self.get_dimension()
        for i in range(x):
            for j in range(y):
                if (self.matrix[i][j] + value) <= 255:
                    self.matrix[i][j] += value
                else: self.matrix[i][j] = 255

    #make the image darker by soustracting the argument value to every pixel
    def darker(self, value):
        (x, y) = self.get_dimension()
        for i in range(x):
            for j in range(y):
                if (self.matrix[i][j] - value) >= 0:
                    self.matrix[i][j] -= value
                else: self.matrix[i][j] = 0    

    #return the average value of pixels in the (x1, x2) (y1, y2) area
    def averagevalue(self, x1, x2, y1, y2):
        total = 0
        for i in range(x1, x2):
            for j in range(y1, y2):
                total += self.matrix[i][j]
        averagevalue = total/((x2-x1)*(y2-y1))
        return(averagevalue)
    
    #for every pixel, take the 8 values around (+its own value) and get the average value and apply it to the pixel
    #new image is stored into title
    #title needs to be "title.png"
    #function doesn't modify self.matrix
    #WARNING : takes around a min for 1000*1000 pixel image with a value of 10
    #its enough, already a lot smoother with 10
    #NO NEED TO COMPACT IT, and its hard due to how matrice works in np
    def smoothering(self, title, squaresize):
        smooth_matrix = self.matrix
        (x, y) = self.get_dimension()
        #corner1
        for i in range(0, squaresize):
            for j in range(0, squaresize):
                a = self.averagevalue(0, i+squaresize+1, 0, j+squaresize+1)
                smooth_matrix[i][j] = a
        #corner2
        for i in range(x-squaresize, x):
            for j in range(y-squaresize, y):
                a = self.averagevalue(i-squaresize, x, j-squaresize, y)
                smooth_matrix[i][j] = a
        #corner3
        for i in range(x-squaresize, x):
            for j in range(0, squaresize):
                a = self.averagevalue(i-squaresize, x, j, j+squaresize+1)
                smooth_matrix[i][j] = a
        #corner4
        for i in range(0, squaresize):
            for j in range(y-squaresize, y):
                a = self.averagevalue(0, i+squaresize+1, j-squaresize, y)
                smooth_matrix[i][j] = a
        #center
        for i in range(squaresize, x-squaresize):
            for j in range(squaresize, y-squaresize):
                a = self.averagevalue(i-squaresize, i+squaresize+1, j-squaresize, j+squaresize+1)
                smooth_matrix[i][j] = a
        #strip1
        for i in range(squaresize, x-squaresize):
            for j in range(0, squaresize):
                a = self.averagevalue(i-squaresize, i+squaresize+1, 0, j+squaresize+1)
                smooth_matrix[i][j] = a
        #strip2
        for i in range(squaresize, x-squaresize):
            for j in range(y-squaresize, y):
                a = self.averagevalue(i-squaresize, i+squaresize+1, j-squaresize, y)
                smooth_matrix[i][j] = a
        #strip3
        for i in range(0, squaresize):
            for j in range(squaresize, y-squaresize):
                a = self.averagevalue(0, i+squaresize+1, j-squaresize, j+squaresize+1)
                smooth_matrix[i][j] = a
        #strip4
        for i in range(x-squaresize, x):
            for j in range(squaresize, y-squaresize):
                a = self.averagevalue(x-squaresize, x, j-squaresize, j+squaresize+1)
                smooth_matrix[i][j] = a
        cv2.imwrite(title, smooth_matrix)


a = Matrix("test.jpg")
a.smoothering("test01.png", 5)
print("done")
