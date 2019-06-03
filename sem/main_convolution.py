from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sem.net_convolution import CNN


# img parameters
train_path = 'img/train'
test_path = 'img/test'
height = 28
width = 28
depth = 3
bits = 8

classes = 10 # count of result classes

batch_size = 30 # learning batch
epochs = 1000

net_c = CNN(32, (3, 3), (height, width, depth), (1, 1), 2, 3)

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
out = net_c.evaluate_generator(test_generator, 1000)
print(out)
