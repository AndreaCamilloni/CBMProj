import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Activation, Conv2D, MaxPool2D, Flatten

from keras.regularizers import l2
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)


def single_task_model():
    base_model = keras.applications.InceptionV3(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(256, 256, 3),  # VGG16 expects min 32 x 32
        include_top=False)  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False
    inputs = keras.Input(shape=(256, 256, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    activation = None  # tf.keras.activations.sigmoid or softmax

    outputs = keras.layers.Dense(2,
                                 kernel_initializer=initializer,
                                 activation=activation,
                                 name='predictions')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss={
            'predictions': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=tf.keras.metrics.CategoricalAccuracy()
    )
    return model


def Model_XtoC(loss_list, test_metrics, dd):
    base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False)

    # freeze all the layers
    for layer in base_model.layers[:]:
        layer.trainable = False

    model_input = Input(shape=(224, 224, 3))
    x = base_model(model_input)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    x = Dropout(dd)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dd)(x)
    # start passing that fully connected block output to all the
    # different model heads
    y1 = Dense(128, activation='relu')(x)
    y1 = Dropout(dd)(y1)
    y1 = Dense(64, activation='relu')(y1)
    y1 = Dropout(dd)(y1)

    y2 = Dense(128, activation='relu')(x)
    y2 = Dropout(dd)(y2)
    y2 = Dense(64, activation='relu')(y2)
    y2 = Dropout(dd)(y2)

    y3 = Dense(128, activation='relu')(x)
    y3 = Dropout(dd)(y3)
    y3 = Dense(64, activation='relu')(y3)
    y3 = Dropout(dd)(y3)

    y4 = Dense(128, activation='relu')(x)
    y4 = Dropout(dd)(y4)
    y4 = Dense(64, activation='relu')(y4)
    y4 = Dropout(dd)(y4)

    y5 = Dense(128, activation='relu')(x)
    y5 = Dropout(dd)(y5)
    y5 = Dense(64, activation='relu')(y5)
    y5 = Dropout(dd)(y5)

    y6 = Dense(128, activation='relu')(x)
    y6 = Dropout(dd)(y6)
    y6 = Dense(64, activation='relu')(y6)
    y6 = Dropout(dd)(y6)

    y7 = Dense(128, activation='relu')(x)
    y7 = Dropout(dd)(y7)
    y7 = Dense(64, activation='relu')(y7)
    y7 = Dropout(dd)(y7)

    # connect all the heads to their final output layers
    y1 = Dense(3, activation='sigmoid', name='pigment_network_numeric')(y1)
    y2 = Dense(2, activation='sigmoid', name='blue_whitish_veil_numeric')(y2)
    y3 = Dense(3, activation='sigmoid', name='vascular_structures_numeric')(y3)
    y4 = Dense(3, activation='sigmoid', name='pigmentation_numeric')(y4)
    y5 = Dense(3, activation='sigmoid', name='streaks_numeric')(y5)
    y6 = Dense(3, activation='sigmoid', name='dots_and_globules_numeric')(y6)
    y7 = Dense(2, activation='sigmoid', name='regression_structures_numeric')(y7)

    model = Model(inputs=model_input, outputs=[y1, y2, y3, y4, y5, y6, y7])

    model.compile(loss=loss_list, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), metrics=test_metrics)

    return model


def multi_task_model():
    base_model = keras.applications.ResNet50(weights=None, classes=14, include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    act_0 = Activation('relu', name='act_0')(x)

    # Task 1
    conv_1 = Conv2D(32, 3, name='conv_1')(act_0)
    act_1 = Activation('relu', name='act_1')(conv_1)
    pool_1 = MaxPool2D(4, name='pool_1')(act_1)
    flat_1 = Flatten(name='flat_1')(pool_1)

    t1 = Dense(2, activation='sigmoid', name='t1')(flat_1)

    # Task 2
    conv_2 = Conv2D(32, 3, name='conv_2')(act_0)
    act_2 = Activation('relu', name='act_2')(conv_2)
    pool_2 = MaxPool2D(4, name='pool_2')(act_2)
    flat_2 = Flatten(name='flat_2')(pool_2)

    t2 = Dense(2, activation='sigmoid', name='t2')(flat_2)

    # Task 3
    conv_3 = Conv2D(32, 3, name='conv_3')(act_0)
    act_3 = Activation('relu', name='act_3')(conv_3)
    pool_3 = MaxPool2D(4, name='pool_3')(act_3)
    flat_3 = Flatten(name='flat_3')(pool_3)

    t3 = Dense(2, activation='sigmoid', name='t3')(flat_3)

    # Task 4
    conv_4 = Conv2D(32, 3, name='conv_4')(act_0)
    act_4 = Activation('relu', name='act_4')(conv_4)
    pool_4 = MaxPool2D(4, name='pool_4')(act_4)
    flat_4 = Flatten(name='flat_4')(pool_4)

    t4 = Dense(2, activation='sigmoid', name='t4')(flat_4)

    # Task 5
    conv_5 = Conv2D(32, 3, name='conv_5')(act_0)
    act_5 = Activation('relu', name='act_5')(conv_5)
    pool_5 = MaxPool2D(4, name='pool_5')(act_5)
    flat_5 = Flatten(name='flat_5')(pool_5)

    t5 = Dense(2, activation='sigmoid', name='t5')(flat_5)

    # Task 6
    conv_6 = Conv2D(32, 3, name='conv_6')(act_0)
    act_6 = Activation('relu', name='act_6')(conv_6)
    pool_6 = MaxPool2D(4, name='pool_6')(act_6)
    flat_6 = Flatten(name='flat_6')(pool_6)

    t6 = Dense(2, activation='sigmoid', name='t6')(flat_6)

    # Task 7
    conv_7 = Conv2D(32, 3, name='conv_7')(act_0)
    act_7 = Activation('relu', name='act_7')(conv_7)
    pool_7 = MaxPool2D(4, name='pool_7')(act_7)
    flat_7 = Flatten(name='flat_7')(pool_7)

    t7 = Dense(2, activation='sigmoid', name='t7')(flat_7)

    # Task DIAGNOSIS
    conv_diag = Conv2D(32, 3, name='conv_diag')(act_0)
    act_diag = Activation('relu', name='act_diag')(conv_diag)
    pool_diag = MaxPool2D(4, name='pool_diag')(act_diag)
    flat_diag = Flatten(name='flat_diag')(pool_diag)

    diag = Dense(2, activation='sigmoid', name='diag')(flat_diag)

    # Groups layer in Model object
    model = tf.keras.models.Model(base_model.input, [t1, t2, t3, t4, t5, t6, t7, diag])

    # we can define a multiple loss by dict
    # key = name of dense layer
    model.compile(
        loss={
            't1': 'binary_crossentropy',
            't2': 'binary_crossentropy',
            't3': 'binary_crossentropy',
            't4': 'binary_crossentropy',
            't5': 'binary_crossentropy',
            't6': 'binary_crossentropy',
            't7': 'binary_crossentropy',
            'diag': 'binary_crossentropy'
        },
        optimizer='adam',
        metrics=['accuracy']

    )

    return model
