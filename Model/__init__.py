import tf as tf
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Activation, Conv2D, MaxPool2D, Flatten


def single_task_model():
    base_model = keras.applications.ResNet50(weights=None, classes=2, include_top=False,
                                             input_shape=(256,256,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax', name='predictions')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        loss={
            'predictions': 'binary_crossentropy'
        },
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def multi_task_model():
    input_ = Input(shape=(256, 256, 3), name='input')  # Input tensor 256x256x3
    conv_0 = Conv2D(32, 3, name='conv_0')(input_)
    act_0 = Activation('relu', name='act_0')(conv_0)

    # Task 1
    conv_1 = Conv2D(32, 3, name='conv_1')(act_0)
    act_1 = Activation('relu', name='act_1')(conv_1)
    pool_1 = MaxPool2D(4, name='pool_1')(act_1)
    flat_1 = Flatten(name='flat_1')(pool_1)

    t1 = Dense(1, activation='sigmoid', name='t1')(flat_1)

    # Task 2
    conv_2 = Conv2D(32, 3, name='conv_2')(act_0)
    act_2 = Activation('relu', name='act_2')(conv_2)
    pool_2 = MaxPool2D(4, name='pool_2')(act_2)
    flat_2 = Flatten(name='flat_2')(pool_2)

    t2 = Dense(1, activation='sigmoid', name='t2')(flat_2)

    # Task 3
    conv_3 = Conv2D(32, 3, name='conv_3')(act_0)
    act_3 = Activation('relu', name='act_3')(conv_3)
    pool_3 = MaxPool2D(4, name='pool_3')(act_3)
    flat_3 = Flatten(name='flat_3')(pool_3)

    t3 = Dense(1, activation='sigmoid', name='t3')(flat_3)

    # Task 4
    conv_4 = Conv2D(32, 3, name='conv_4')(act_0)
    act_4 = Activation('relu', name='act_4')(conv_4)
    pool_4 = MaxPool2D(4, name='pool_4')(act_4)
    flat_4 = Flatten(name='flat_4')(pool_4)

    t4 = Dense(1, activation='sigmoid', name='t4')(flat_4)

    # Task 5
    conv_5 = Conv2D(32, 3, name='conv_5')(act_0)
    act_5 = Activation('relu', name='act_5')(conv_5)
    pool_5 = MaxPool2D(4, name='pool_5')(act_5)
    flat_5 = Flatten(name='flat_5')(pool_5)

    t5 = Dense(1, activation='sigmoid', name='t5')(flat_5)

    # Task 6
    conv_6 = Conv2D(32, 3, name='conv_6')(act_0)
    act_6 = Activation('relu', name='act_6')(conv_6)
    pool_6 = MaxPool2D(4, name='pool_6')(act_6)
    flat_6 = Flatten(name='flat_6')(pool_6)

    t6 = Dense(1, activation='sigmoid', name='t6')(flat_6)

    # Task 7
    conv_7 = Conv2D(32, 3, name='conv_7')(act_0)
    act_7 = Activation('relu', name='act_7')(conv_7)
    pool_7 = MaxPool2D(4, name='pool_7')(act_7)
    flat_7 = Flatten(name='flat_7')(pool_7)

    t7 = Dense(1, activation='sigmoid', name='t7')(flat_7)

    # Task DIAGNOSIS
    conv_diag = Conv2D(32, 3, name='conv_diag')(act_0)
    act_diag = Activation('relu', name='act_diag')(conv_diag)
    pool_diag = MaxPool2D(4, name='pool_diag')(act_diag)
    flat_diag = Flatten(name='flat_diag')(pool_diag)

    diag = Dense(1, activation='sigmoid', name='diag')(flat_diag)

    # Groups layer in Model object
    model = tf.keras.models.Model(input_, [t1, t2, t3, t4, t5, t6, t7, diag])

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