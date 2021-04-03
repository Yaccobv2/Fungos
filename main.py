# TODO:
#  add graphs acc(epoch), loss(epoch)

import functions
import numpy as np
import tensorflow as tf
import itertools

#dir = "E:/GitHub/fungos_libs/Mushrooms"
dir = "C:/Users/Mateusz/Documents/GitHub/fungos_dataset/Mushrooms"

#######################################
batch_size = 50
epochs = 80
img_size = (50, 50, 3)
imgs_per_class = 400
#######################################

images, labels, classes = functions.readData(dir,imgs_per_class)

print("Number of classes: ", len(classes))
print("Number of images: ", len(images))
print("Number of labels: ", len(labels))
print("clases: :", classes)
print("labels: ", labels)

images = functions.createNpArray(images)
classes = functions.createNpArray(classes)

print(images[0].shape)
images = list(map(functions.resize, images, itertools.repeat((img_size[0], img_size[1]), len(images))))
print(images[0].shape)

train_data, test_data, train_label, test_label = functions.splitDataIntoContainers(images, labels, 0.2)
train_data, validation_data, train_label, validation_label = functions.splitDataIntoContainers(train_data, train_label,
                                                                                               0.1)

print("Number of training images: ", len(train_data))
print("Number of testing images: ", len(test_data))
print("Number f validating data: ", len(validation_data))

# prepare labels to create output layer
train_label = functions.oneHotEncoding(train_label, len(classes))
test_label = functions.oneHotEncoding(test_label, len(classes))
validation_label = functions.oneHotEncoding(validation_label, len(classes))

# data augmentation
dataGen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.1,
                             shear_range=0.2,
                             rotation_range=20)
dataGen.fit(train_data)

# create neural network
model = functions.createNeuralNetwork((img_size[1], img_size[0], img_size[2]), len(classes))

# print(model.summary())
print("########")
print(len(train_data))
print(len(train_label))
print("########")


# train model
x = model.fit(dataGen.flow(np.array(train_data), np.array(train_label), batch_size=batch_size), epochs=epochs, steps_per_epoch=len(train_data)/batch_size,
              validation_data=(np.array(validation_data), np.array(validation_label)), shuffle=True)

# check accuracy on test data
y = model.evaluate(np.array(test_data), np.array(test_label))


# print results
print("Funkcja bledu: ", y[0])
print("Skutecznosc: ", y[1])

# save model
name = 'model_1_'+str(img_size[0])+'x'+str(img_size[1])+'_batch'+str(batch_size)+'_epochs'+str(epochs)+'_imgs'+str(imgs_per_class)+'_acc'+str(round(y[1], 2))+'.h5'
print(name)
model.save(name)
