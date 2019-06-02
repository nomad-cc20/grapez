from sem.image_preprocessor import ImagePreProcessor
from sem.net_Kohonen_map import Kohonen
import numpy as np
import os
from tqdm import tqdm

path = 'img/'
testFolder = 'test/'
trainFolder = 'train/'
folders = {'pos', 'neg', 'env'}

ipp = ImagePreProcessor(8, 2)
length = 0
patternCount = 0
for folderName in os.listdir(path + trainFolder):
    for fileName in os.listdir(path + trainFolder + '/' + folderName):
        img = ipp.serve(path + trainFolder + '/' + folderName + '/' + fileName)
        if len(img) > length:
            length = len(img)
        patternCount = patternCount + 1

patterns = np.zeros((patternCount, length))
i = 0
for folderName in os.listdir(path + trainFolder):
    for fileName in os.listdir(path + trainFolder + '/' + folderName):
        img = ipp.serve(path + trainFolder + '/' + folderName + '/' + fileName)
        patterns[i] = img
        i = i + 1
patterns = np.transpose(patterns)

net_km = Kohonen(3, length)
net_km.train(patterns, 0.9, 0.995, 5, 1)

i = 0
results = np.zeros((3, 200))
for folderName in os.listdir(path + testFolder):
    j = 0
    for fileName in tqdm(os.listdir(path + testFolder + '/' + folderName)):
        img = ipp.serve(path + testFolder + '/' + folderName + '/' + fileName)
        results[i, j] = net_km.equip(img)[0]
        j = j + 1
    i = i + 1

print(results)
