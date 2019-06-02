import os

import matplotlib.pyplot as plt
import numpy as np

from sem.image_preprocessor import ImagePreProcessor
from sem.net_FF_Keras import FFNN

path = 'img/'
testFolder = 'test/'
trainFolder = 'train/'
folders = {'pos', 'neg', 'env'}

ipp = ImagePreProcessor(8, 2)
clusters_train = np.zeros(3)
clusters_val = np.zeros(3)
cellCount = 0

i = 0
for folderName in os.listdir(path + trainFolder):
    for fileName in os.listdir(path + trainFolder + '/' + folderName):
        img = ipp.serve(path + trainFolder + '/' + folderName + '/' + fileName)
        if len(img) > cellCount:
            cellCount = len(img)
        clusters_train[i] = clusters_train[i] + 1
    i = i + 1

i = 0
for folderName in os.listdir(path + testFolder):
    for fileName in os.listdir(path + testFolder + '/' + folderName):
        img = ipp.serve(path + testFolder + '/' + folderName + '/' + fileName)
        if len(img) > cellCount:
            cellCount = len(img)
        clusters_val[i] = clusters_val[i] + 1
    i = i + 1

patterns_train = np.zeros((clusters_train[0] + clusters_train[1] + clusters_train[2], cellCount))
i = 0
for folderName in os.listdir(path + trainFolder):
    for fileName in os.listdir(path + trainFolder + '/' + folderName):
        img = ipp.serve(path + trainFolder + '/' + folderName + '/' + fileName)
        patterns_train[i] = img
        i = i + 1
np.random.shuffle(patterns_train)
patterns_train = np.transpose(patterns_train)

patterns_val = np.zeros((clusters_val[0] + clusters_val[1] + clusters_val[2], cellCount))
i = 0
for folderName in os.listdir(path + trainFolder):
    for fileName in os.listdir(path + trainFolder + '/' + folderName):
        img = ipp.serve(path + trainFolder + '/' + folderName + '/' + fileName)
        patterns_val[i] = img
        i = i + 1
np.random.shuffle(patterns_val)
patterns_val = np.transpose(patterns_val)

expected_train = np.zeros(clusters_train[0] + clusters_train[1] + clusters_train[2])
for i in range(clusters_train[0], clusters_train[1]):
    expected_train[i] = 1
for i in range(clusters_train[1], clusters_train[2]):
    expected_train[i] = 2

expected_val = np.zeros(clusters_val[0] + clusters_val[1] + clusters_val[2])
for i in range(clusters_val[0], clusters_val[1]):
    expected_val[i] = 1
for i in range(clusters_val[1], clusters_val[2]):
    expected_val[i] = 2

net_ff = FFNN(cellCount, 6, 3, 1)
epochs = 500

net_ff.fit(patterns_train, expected_train, epochs)

predictions = net_ff.predict(patterns_val)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(patterns_train[0, :], patterns_train[1, :], predictions)
ax.scatter3D(patterns_train[0, :], patterns_train[1, :], expected_val, cmap='Greens')
plt.show()
