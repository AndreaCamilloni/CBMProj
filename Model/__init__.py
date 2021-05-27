import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Activation, Conv2D, MaxPool2D, Flatten

from keras.regularizers import l2
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from Utils.focal_xentropy import FocalCategoricalCrossEntropy


def single_task_model(lr=0.01, input_shape=(256, 256, 3), base='IncNet', loss = "CatXentropy"):
    if base == 'IncNet':
        base_model = keras.applications.InceptionV3(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=input_shape,  # VGG16 expects min 32 x 32
            include_top=False)  # Do not include the ImageNet classifier at the top.
    elif base == 'MobNet':
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=input_shape,  # VGG16 expects min 32 x 32
            include_top=False)  # Do not include the ImageNet classifier at the top.
    elif base == 'ResNet':
        base_model = keras.applications.ResNet101V2(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=input_shape,  # VGG16 expects min 32 x 32
            include_top=False)  # Do not include the ImageNet classifier at the top.

    if loss == 'CatXentropy':
        loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        loss_function = FocalCategoricalCrossEntropy(beta=2., from_logits=True)

    base_model.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    activation = None  # tf.keras.activations.sigmoid or softmax

    outputs = keras.layers.Dense(2,
                                 kernel_initializer=initializer,
                                 activation=activation,
                                 name='predictions')(x)

    model = keras.Model(inputs, outputs)

    #opt = keras.optimizers.Adam(learning_rate=lr)
    opt = keras.optimizers.SGD(learning_rate=lr)
    model.compile(
        loss={
            'predictions': loss_function
        },
        optimizer=opt,
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

