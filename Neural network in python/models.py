import tensorflow as tf


def model_1(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
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

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                    bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.7))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model


def model_3(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(100, activation='relu'))

    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model


def model_4(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, 7,strides=1, activation='relu',kernel_initializer='he_normal', padding='same',input_shape=input_shape))
    model.add(tf.keras.layers.AveragePooling2D(2))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GaussianNoise(0.1))

    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.AveragePooling2D(2))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GaussianNoise(0.1))

    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))  
    model.add(tf.keras.layers.AveragePooling2D(2))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GaussianNoise(0.1))
    
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GaussianNoise(0.1))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))


    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model


def model_5(input_shape, number_of_classes):
    class ResidualUnit(tf.keras.layers.Layer):
        def __init__(self, filters, strides=1, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.activation = tf.keras.activations.get(activation)
            self.main_leyers = [
                tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                self.activation,
                tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization()]
            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                    tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization()]

            def call(self, inputs):
                Z = inputs
                for layer in self.main_leyers:
                    Z = layer(Z)
                skip_Z = inputs
                for layer in self.skip_layers:
                    skip_Z = layer(skip_Z)
                return self.activation(Z + skip_Z)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, 7, strides=2,input_shape=input_shape, padding="same", use_bias=False))
    tf.keras.layers.BatchNormalization()
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters,strides=strides))
        prev_filters = filters
    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation="softmax"))
    
    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    return model

def model_7(input_shape, number_of_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, 7,strides=1, activation='relu',kernel_initializer='he_normal', padding='same',input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                     bias_regularizer=tf.keras.regularizers.l2(0.01)))  
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics='accuracy')
    print(model.summary())
    return model

# model_1((200, 200, 3), 9)
# model_2((100, 100, 3), 9)
