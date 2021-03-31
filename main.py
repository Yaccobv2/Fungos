import functions
import numpy as np

dir = "E:/GitHub/fungos_libs/Mushrooms"


images, labels, classes = functions.readData(dir,500)

print("Number of classes: ", len(classes))
print("Number of images: ", len(images))
print("Number of labels: ", len(labels))
print("clases: :", classes)
print("labels: ", labels)

images = functions.createNpArray(images)
classes = functions.createNpArray(classes)

print(images[0].shape)
images = list(map(functions.resize, images))
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

# create neural network
model = functions.createNeuralNetwork((50, 50, 3), len(classes))

# print(model.summary())
print("########")
print(len(train_data))
print(len(train_label))
print("########")

x = model.fit(np.array(train_data), np.array(train_label), batch_size=60, epochs=30, steps_per_epoch=20,
              validation_data=(np.array(validation_data), np.array(validation_label)), shuffle=True)

y = model.evaluate(np.array(test_data), np.array(test_label))

print("Funkcja bledu: ", y[0])
print("Skutecznosc: ", y[1])
