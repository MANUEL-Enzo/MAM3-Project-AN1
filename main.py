import math as mt
import numpy as np
import matplotlib.pyplot as plt

Q = [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 13, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]]


def P():
    p = []

    for j in range(8):
        py = []

        for i in range(8):
            py.append(0.5 * (mt.sqrt(1 / 2) if (j == 0) else 1) * mt.cos((2 * i + 1) * j * mt.pi / 16))

        p.append(py)

    return p


def pictureToRGBList(image):
    height = len(image) - (len(image) % 8)
    width = len(image[0]) - (len(image[0]) % 8)

    matrixListR = []
    matrixListG = []
    matrixListB = []

    for x in range(int(height / 8)):
        for y in range(int(width / 8)):
            tempR = []
            tempG = []
            tempB = []

            for xt in range(8):
                tempYR = []
                tempYG = []
                tempYB = []

                for yt in range(8):
                    tempYR.append(image[x * 8 + xt][y * 8 + yt][0] - 128)
                    tempYG.append(image[x * 8 + xt][y * 8 + yt][1] - 128)
                    tempYB.append(image[x * 8 + xt][y * 8 + yt][2] - 128)

                tempR.append(tempYR)
                tempG.append(tempYG)
                tempB.append(tempYB)

            matrixListR.append(tempR)
            matrixListG.append(tempG)
            matrixListB.append(tempB)

    return matrixListR, matrixListB, matrixListG


def compression(ms, p):
    cs = []

    for m in ms:
        cs.append(matrixCompression(m, p))

    return cs


def matrixCompression(m, p):
    d = np.matmul(p, np.matmul(m, np.transpose(p)))

    c = np.trunc(np.divide(d, Q))

    x = int(input("Valeur de filtrage :"))

    for i in range(8):
        for j in range(8):
            if i + j >= x:
                c[i][j] = 0

    return c


image_file = "5.jpg"

image = plt.imread(image_file)

matrixListR, matrixListG, matrixListB = pictureToRGBList(image)

matrixListCompressionR = compression(matrixListR, P())
matrixListCompressionG = compression(matrixListG, P())
matrixListCompressionB = compression(matrixListB, P())


