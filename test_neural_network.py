# TODO
#   add testing multiple images
#   add confusion matrix

import tensorflow as tf
import functions as fc
import numpy as np
from sklearn.metrics import confusion_matrix

# load image
#dir = "C:/Users/Mateusz/Documents/GitHub/fungos_dataset/uporzadkowane/0/352_NTs98vtOSGA.jpg"
#img = fc.loadForPrediction(dir, 50, 50)

# load model
#model = tf.keras.models.load_model('model_1_50x50_batch50_epochs80_imgs400_acc0.48.h5')
#print(model.summary())

# predict
#print(model.predict(img))

def test(model, imgs, labels):
    imgs_reshape = []
    for img in imgs:
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        imgs_reshape.append(img)
    preds = []
    labels = np.argmax(labels, axis=1)
    for img in imgs_reshape:
        preds.append(np.argmax(model.predict(img), axis=1))
        model.predict_classes(img)

    M = confusion_matrix(labels, preds)
    print(M)
    
    #for column in M.T:
        #some_function(column)




