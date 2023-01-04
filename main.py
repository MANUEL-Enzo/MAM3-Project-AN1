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


def matrixListToMatrix(ListR, ListG, ListB, height, width, offset=0, norm=1):
    Mat = np.empty((height, width, 3))

    for a in range(int(width/8)):
        for b in range(int(height/8)):
            for i in range(8):
                for j in range(8):
                    Mat[b*8+j][a*8+i] = (
                        [(ListR[b*int(width/8)+a][i][j] + offset)/norm,
                         (ListG[b*int(width/8)+a][i][j] + offset)/norm,
                         (ListB[b*int(width/8)+a][i][j] + offset)/norm])

    return Mat


def matrixListToPicture(ListR, ListG, ListB, height, width):
    return matrixListToMatrix(ListR, ListG, ListB, height, width, 128, 256)


def compression(ms, p, x):
    cs = []

    for m in ms:
        cs.append(matrixCompression(m, p, x))

    return cs


def matrixCompression(m, p, x):
    d = np.matmul(p, np.matmul(m, np.transpose(p)))

    c = np.trunc(np.divide(d, Q))

    for i in range(8):
        for j in range(8):
            if i + j >= x:
                c[i][j] = 0

    return c


def decompression(ms, p):
    cs = []

    for m in ms:
        cs.append(matrixDecompression(m, p))

    return cs


def matrixDecompression(m, p):
    d = np.trunc(np.multiply(m, Q))

    return np.matmul(np.transpose(p), np.matmul(d, p))


def comparaison(a,b):
    diffTermeATermeR = 0
    diffTermeATermeG = 0
    diffTermeATermeB = 0

    c, d, e = min(a.shape, b.shape)

    aR = np.empty((c, d))
    aG = np.empty((c, d))
    aB = np.empty((c, d))
    bR = np.empty((c, d))
    bG = np.empty((c, d))
    bB = np.empty((c, d))

    for i in range(c):
        for j in range(d):
            aR[i, j] = a[i, j][0]
            aB[i, j] = a[i, j][1]
            aG[i, j] = a[i, j][2]
            bR[i, j] = b[i, j][0]
            bB[i, j] = b[i, j][1]
            bG[i, j] = b[i, j][2]

    for i in range(c):
        for j in range(d):
            diffTermeATermeR += abs(aR[i, j]-bR[i, j])
            diffTermeATermeG += abs(aB[i, j]-bB[i, j])
            diffTermeATermeB += abs(aG[i, j]-bG[i, j])

    normAR = np.linalg.norm(aR, ord=2)
    normAG = np.linalg.norm(aG, ord=2)
    normAB = np.linalg.norm(aB, ord=2)
    normBR = np.linalg.norm(bR, ord=2)
    normBG = np.linalg.norm(bG, ord=2)
    normBB = np.linalg.norm(bB, ord=2)

    diffNormR = normAR / normBR
    diffNormG = normAG / normBG
    diffNormB = normAB / normBB

    return [[diffTermeATermeR, diffTermeATermeR, diffTermeATermeR], [diffNormR, diffNormG, diffNormB]]


image_file = "5.jpg"

image = plt.imread(image_file)

height = len(image) - (len(image) % 8)
width = len(image[0]) - (len(image[0]) % 8)

matrixListR, matrixListG, matrixListB = pictureToMatrixList(image)

x = int(input("Valeur de filtrage : "))

matrixListCompressionR = compression(matrixListR, P(), x)
matrixListCompressionG = compression(matrixListG, P(), x)
matrixListCompressionB = compression(matrixListB, P(), x)

compressed = matrixListToMatrix(matrixListCompressionR, matrixListCompressionG, matrixListCompressionB, height, width)

matrixListCompressionR, matrixListCompressionG, matrixListCompressionB = matrixToMatrixList(compressed)

matrixListDecompressionR = decompression(matrixListCompressionR, P())
matrixListDecompressionG = decompression(matrixListCompressionG, P())
matrixListDecompressionB = decompression(matrixListCompressionB, P())

imageDecompressed = matrixListToPicture(matrixListDecompressionR, matrixListDecompressionG, matrixListDecompressionB, height, width)

imgplot = plt.imshow(imageDecompressed)
plt.show()

print(comparaison(image, imageDecompressed))
