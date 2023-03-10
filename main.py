import math as mt
import time
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
    p = np.empty((8, 8))

    for j in range(8):
        for i in range(8):
            p[j, i] = 0.5 * (mt.sqrt(1 / 2) if (j == 0) else 1) * mt.cos((2 * i + 1) * j * mt.pi / 16)

    return p


def matrixToMatrixList(image, offset=0):
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
                    tempYR.append(image[x * 8 + xt][y * 8 + yt][0] - offset)
                    tempYG.append(image[x * 8 + xt][y * 8 + yt][1] - offset)
                    tempYB.append(image[x * 8 + xt][y * 8 + yt][2] - offset)

                tempR.append(tempYR)
                tempG.append(tempYG)
                tempB.append(tempYB)

            matrixListR.append(tempR)
            matrixListG.append(tempG)
            matrixListB.append(tempB)

    return matrixListR, matrixListB, matrixListG


def pictureToMatrixList(image):
    return matrixToMatrixList(image, 128)


def matrixListToMatrix(ListR, ListG, ListB, height, width, offset=0):
    Mat = np.empty((height, width, 3))

    for a in range(int(width / 8)):
        for b in range(int(height / 8)):
            for i in range(8):
                for j in range(8):
                    Mat[b * 8 + j][a * 8 + i] = (
                        [((ListR[int(b * int(width / 8)) + a][i][j]) + offset),
                         ((ListG[int(b * int(width / 8)) + a][i][j]) + offset),
                         ((ListB[int(b * int(width / 8)) + a][i][j]) + offset)])

    return Mat


def matrixListToPicture(ListR, ListG, ListB, height, width):
    return matrixListToMatrix(ListR, ListG, ListB, height, width, 128)


def compression(ms, p, x):
    cs = []

    for m in ms:
        cs.append(matrixCompression(m, p, x))

    return cs


def matrixCompression(m, p, x):
    d = np.matmul(p, np.matmul(m, np.transpose(p)))

    c = np.divide(d, Q)

    for i in range(8):
        for j in range(8):
            if i + j >= x:
                c[i][j] = 0

    return c.astype(int)


def decompression(ms, p):
    cs = []

    for m in ms:
        cs.append(matrixDecompression(m, p))

    return cs


def matrixDecompression(m, p):
    d = np.multiply(m, Q)

    return np.matmul(np.transpose(p), np.matmul(d, p))


def pictureCompression(image, filtration):
    height = len(image) - (len(image) % 8)
    width = len(image[0]) - (len(image[0]) % 8)

    matrixListR, matrixListG, matrixListB = pictureToMatrixList(image)

    return matrixListToMatrix(compression(matrixListR, P(), filtration), compression(matrixListG, P(), filtration), compression(matrixListB, P(), filtration), height, width).astype(int)


def pictureDecompression(compressed):
    height = len(compressed)
    width = len(compressed[0])

    matrixListCompressionR, matrixListCompressionG, matrixListCompressionB = matrixToMatrixList(compressed)

    return matrixListToPicture(decompression(matrixListCompressionR, P()), decompression(matrixListCompressionG, P()), decompression(matrixListCompressionB, P()), height, width).astype(int)


def comparaison(a, b):
    c, d, e = min(a.shape, b.shape)

    normR = np.linalg.norm(a[:c, :d, 0] - b[:c, :d, 0], ord=2)
    normG = np.linalg.norm(a[:c, :d, 1] - b[:c, :d, 1], ord=2)
    normB = np.linalg.norm(a[:c, :d, 2] - b[:c, :d, 2], ord=2)

    diffNormR = normR / np.linalg.norm(a[:c, :d, 0], ord=2)
    diffNormG = normG / np.linalg.norm(a[:c, :d, 1], ord=2)
    diffNormB = normB / np.linalg.norm(a[:c, :d, 2], ord=2)

    return [diffNormR*100, diffNormG*100, diffNormB*100]


image_file = "1.jpg"

image = plt.imread(image_file)

filtration = 15

tic = time.time()

compressed = pictureCompression(image, filtration)

imageDecompressed = pictureDecompression(compressed)

toc = time.time() - tic

print("Valeur de filtrage : ", filtration,
      " Pourcentage de z??ros : ", (1-np.count_nonzero(compressed.astype(int))/(len(compressed)*len(compressed[0])))*100,
      " Pourcentage d'erreur [R, G, B] : ", comparaison(image, imageDecompressed),
      " Temps d'??x??cution : ", toc)