import tensorflow as tf

def model_1(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(100, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model

def model_2(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model

def model_3(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(100, activation='relu'))

    #model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model


def model_4(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape, padding='same'))

    model.add(tf.keras.layers.Conv2D(64, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model



#model_1((200, 200, 3), 9)
#model_2((100, 100, 3), 9)
