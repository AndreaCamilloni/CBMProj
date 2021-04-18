from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D


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
