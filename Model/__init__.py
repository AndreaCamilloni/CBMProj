from tensorflow import keras
from tensorflow.python.keras import Input


def single_task_model():
    model = keras.applications.ResNet50(weights=None, classes=2,input_shape=(256,256,3))
    model.compile(
        loss={
            'predictions': 'binary_crossentropy'
        },
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
