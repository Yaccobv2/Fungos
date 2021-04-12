import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tensorflow as tf


def resize(img, size):
    img = cv2.resize(img, (size[0], size[1]))
    # plt.imshow(img)
    return img.astype(np.float32)/255


def readData(dir,number_of_images):
    images = []
    labels = []
    classes = []
    print("Reading data...")
    for folder in os.listdir(dir):
        counter = 0
        for imgs in os.listdir(dir + "/" + folder):
            img = cv2.imread(dir + "/" + folder + "/" + imgs)

            images.append(img)
            labels.append(str(folder))
            print("len(zdjecia): ", len(images))
            counter += 1
            if folder not in classes:
                classes.append(folder)
            if counter >= number_of_images:
                break
    return images, labels, classes


def createNpArray(input_array):
    output_array = np.array(input_array)
    return output_array


def splitDataIntoContainers(data, labels, size):
    data1, data2, label1, label2 = sklearn.model_selection.train_test_split(data, labels, test_size=size,random_state=42)
    return data1, data2, label1, label2


def oneHotEncoding(labels, number_of_classes):
    encoded_labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    return encoded_labels


def createNeuralNetwork(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model

def loadForPrediction(dir, width, height):
    img = tf.keras.preprocessing.image.load_img(dir, target_size=(width, height))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)/255
    