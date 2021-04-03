# TODO
#   add testing multiple images
#   add confusion matrix

import tensorflow as tf
import functions as fc
import numpy as np

# load image
dir = "C:/Users/Mateusz/Documents/GitHub/fungos_dataset/uporzadkowane/0/352_NTs98vtOSGA.jpg"
img = fc.loadForPrediction(dir, 50, 50)

# load model
model = tf.keras.models.load_model('model_1_50x50_batch50_epochs80_imgs400_acc0.48.h5')
#print(model.summary())

# predict
print(model.predict(img))
