import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib
import sklearn.model_selection

liczba_klas=8
train_data = []
train_label =[]
validation_data = []
validation_label = []
test_data = []
test_label = []

zdjecia = []
labele = []
klasy = []

def resize(img):
    img = cv2.resize(img, (50, 50))
    return img

print("Zaczynam wczytywac")
for folder in os.listdir("C:/Users/Mateusz/Documents/GitHub/Fungos_dataset/Mushrooms"):
    licznik=0
    for imgs in os.listdir("C:/Users/Mateusz/Documents/GitHub/Fungos_dataset/Mushrooms"+"/"+folder):
        img = cv2.imread("C:/Users/Mateusz/Documents/GitHub/Fungos_dataset/Mushrooms"+"/"+folder+"/"+imgs, )

        zdjecia.append(img)
        labele.append(str(folder))
        print("len(zdjecia): ", len(zdjecia))
        licznik += 1
        if folder not in klasy:
            klasy.append(folder)
        if(licznik>=100):
            break


print()

print("Liczba klas: ", len(klasy))
print("Liczba obrazow: ", len(zdjecia))
print("Liczba labeli: ", len(labele))
print()
print("klasy: :", klasy)
print("labele: ", labele)
print()

zdjecia = np.array(zdjecia)
klasy = np.array(klasy)

print()
print(zdjecia[0].shape)
zdjecia = list(map(resize, zdjecia))
print(zdjecia[0].shape)
print()

train_data, test_data, train_label,  test_label = sklearn.model_selection.train_test_split(zdjecia, labele, test_size=0.2)
train_data, validation_data, train_label, validation_label = sklearn.model_selection.train_test_split(train_data, train_label, test_size=0.1)

print("Liczba zdjec treningowych: ", len(train_data))
print("Liczba zdjec testowych: ", len(test_data))
print("Liczba zdjec walidacyjnych: ", len(validation_data))

train_label=tf.keras.utils.to_categorical(train_label, len(klasy))
test_label=tf.keras.utils.to_categorical(test_label, len(klasy))
validation_label=tf.keras.utils.to_categorical(validation_label, len(klasy))

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape= (50, 50, 3)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(len(klasy), activation='softmax'))

model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
#print(model.summary())
print("########")
print(len(train_data))
print(len(train_label))
print("########")

x = model.fit(np.array(train_data), np.array(train_label), batch_size=30, epochs=10, steps_per_epoch=20, validation_data=(np.array(validation_data), np.array(validation_label)), shuffle=True)

y = model.evaluate(np.array(test_data), np.array(test_label))

print("Funkcja bledu: ", y[0])
print("Skutecznosc: ", y[1])