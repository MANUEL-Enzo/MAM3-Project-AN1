import numpy as np
import matplotlib.pyplot as plt

image_file = "5.jpg"

image = plt.imread(image_file)

height = len(image) - (len(image) % 8)
width = len(image[0]) - (len(image[0]) % 8)

listR = []
listG = []
listB = []

for x in range(int(height/8)):
    for y in range(int(width/8)):
        tempR = []
        tempG = []
        tempB = []

        for xt in range(8):
            tempYR = []
            tempYG = []
            tempYB = []

            for yt in range(8):
                tempYR.append(image[x*8+xt][y*8+yt][0] - 128)
                tempYG.append(image[x*8+xt][y*8+yt][1] - 128)
                tempYB.append(image[x*8+xt][y*8+yt][2] - 128)

            tempR.append(tempYR)
            tempG.append(tempYG)
            tempB.append(tempYB)

        listR.append(tempR)
        listG.append(tempG)
        listB.append(tempB)



