# CNN + logisticRegression
import numpy as np
from PIL import Image
from keras.preprocessing import image

from sklearn.linear_model import LogisticRegression


def spatial_average_pooling(x):
    """Average across the entire spatial dimensions."""
    return np.squeeze(x).mean(axis=0).mean(axis=0)


def deep_features(img_paths, model, func_preprocess_input, target_size=(256, 256, 3),
                  func_postprocess_features=spatial_average_pooling):
    """Computes deep features for the images."""

    features = []
    for img_path in img_paths:

        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)
        x = func_preprocess_input(x)
        responses = model.predict(x)

        if func_postprocess_features is not None:
            responses = func_postprocess_features(responses)

        features.append(responses)

    features = np.squeeze(np.asarray(features))

    return features


def logReg(class_weight, train_inputs, train_label, C=0.01):
    return LogisticRegression(C=C, class_weight=class_weight).fit(train_inputs, train_label)


def logReg_predict(reg, test_inputs):
    return reg.predict(test_inputs)
