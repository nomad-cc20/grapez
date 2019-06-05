from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
import numpy as np
import os
from tqdm import tqdm

from sem.net_convolution import CNN


# img parameters
train_path = 'img/train'
test_path = 'img/test'
height = 40
width = 40
depth = 3
bits = 8

classes = 10 # count of result classes

batch_size = 30 # learning batch
epochs = 1000

net_c = CNN(32, (3, 3), (height, width, depth), (1, 1), 1, 3)

# batch operation
train_datagen = ImageDataGenerator(
    rotation_range = 0,
    width_shift_range = 0,
    height_shift_range = 0,
    rescale = 1.0 / (2 ** bits - 1),
    zoom_range = 0,
    fill_mode = 'nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (height, width),
    batch_size = batch_size,
    class_mode = 'categorical'
)
net_c.fit(train_generator, batch_size, epochs)

test_datagen = ImageDataGenerator(
    rotation_range = 0,
    width_shift_range = 0,
    height_shift_range = 0,
    rescale = 1.0 / (2 ** bits - 1),
    zoom_range = 0,
    fill_mode = 'nearest'
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (height, width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

# resulting number of images = steps * batch_size
print(net_c.evaluate(train_generator, 100))
print(net_c.evaluate(test_generator, 100))


path = 'img/'
testFolder = 'test/'
folders = {'pos', 'neg', 'env'}

i = 0
results = np.zeros((3, 200))
for folderName in os.listdir(path + testFolder):
    j = 0
    for fileName in tqdm(os.listdir(path + testFolder + '/' + folderName)):
        results[i, j] = np.argmax(net_c.predict(path + testFolder + '/' + folderName + '/' + fileName, bits))
        #print(fileName)
        #print(net_c.predict(path + testFolder + '/' + folderName + '/' + fileName, bits))
        j = j + 1
    i = i + 1

print(results)
