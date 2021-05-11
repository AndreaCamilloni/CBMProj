import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
from functools import partial


@tf.function
def load_images(x, image_dir):
    image = tf.io.read_file(image_dir + x)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = image / 255.
    # image = imagek.load_img(x)
    # image = imagek.img_to_array(image)
    return image


def load_resize(x,image_dir, new_size=(256, 256)):
    im = load_images(x,image_dir)
    im = tf.image.resize(im, size=new_size)
    return im


def load_aug_resize(x,image_dir, new_size=(256, 256)):
    image = load_images(x,image_dir)
    # augment image
    d1, d2, d3 = image.get_shape()
    r = tf.random.uniform([], minval=0.75, maxval=1., dtype=tf.float32)
    d1, d2 = tf.floor(d1 * r), tf.floor(d2 * r)
    # image = tf.image.random_crop(image, (d1, d2, d3), seed=1993)
    image = tf.image.random_flip_left_right(image, seed=None)
    image = tf.image.random_flip_up_down(image, seed=None)
    image = tf.image.resize(image, new_size)
    return image


class GenericImageSequence(Sequence):
    def __init__(
            self, df: pd.DataFrame, impath_col='derm',
            label_col='diagnosis_numeric', batch_size=1,
            shuffle=False,
            map_fn='resize', one_hot_encoding=True, categories='auto', random_state=None, new_size=(256, 256),
            reshuffle_each_epoch=True, image_dir='/home/andreac/release_v0/images/'
    ):
        super(GenericImageSequence, self).__init__()
        self.image_dir = image_dir
        self.df = df
        self._df = df.copy(deep=True)  # backup copy of original
        self.impath_col = impath_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.random_state = random_state
        self.new_size = new_size
        # pick map fn
        self.update_mapping_fn(map_fn)
        # encoding
        self.one_hot_encoding = one_hot_encoding
        self.categories = categories
        if self.one_hot_encoding:
            self.label_encoder = OneHotEncoder(sparse=False, dtype=np.float32, categories=categories)
        else:
            self.label_encoder = None
        # prepare data
        self.df, self.x, self.y = self.prepare_data(
            self.df, self.shuffle, label_col=self.label_col, impath_col=self.impath_col
        )

    def prepare_data(self, df, shuffle=False, label_col='diagnosis_numeric', impath_col='derm'):
        # shuffle data
        if shuffle:
            df = df.sample(frac=1.)

        # path loader
        x = df[impath_col]

        # encoding
        if self.one_hot_encoding:
            _y = np.reshape(df[label_col].values, (-1, 1))
            y = self.label_encoder.fit_transform(_y)
        else:
            y = df[label_col].values

        return df, x, y

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.x))
        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]
        batch_x = tf.map_fn(self.map_function, batch_x, fn_output_signature=tf.float32)
        return (batch_x, batch_y)

    def on_epoch_end(self):
        if self.reshuffle_each_epoch:
            self.df, self.x, self.y = self.prepare_data(self.df, self.shuffle, self.label_col)

    def update_mapping_fn(self, map_fn, updated_size=None):
        if updated_size is not None:
            print('changing resize size')
            self.new_size = updated_size

        print('updating mapping function')
        if isinstance(map_fn, str):
            if map_fn == 'resize':
                self.map_function = partial(load_resize,image_dir=self.image_dir, new_size=self.new_size)
            elif map_fn == 'aug':
                self.map_function = partial(load_aug_resize,image_dir=self.image_dir, new_size=self.new_size)
            elif map_fn == 'load':
                self.map_function = partial(load_images, image_dir=self.image_dir)
            else:
                raise Exception('no map function defined')
        else:
            self.map_function = map_fn

    def get_all_parameters(self):
        return {
            'derm_col': self.impath_col, 'label_col': self.label_col, 'batch_size': self.batch_size,
            'shuffle': self.shuffle, 'reshuffle_each_epoch': self.reshuffle_each_epoch,
            'random_state': self.random_state, 'one_hot_encoding': self.one_hot_encoding,
            'categories': self.categories, 'new_size': self.new_size
        }


class MultiTaskSequence(Sequence):
    def __init__(
            self, df: pd.DataFrame, impath_col='derm', labels_cols=[
                'pigment_network_numeric',
                'blue_whitish_veil_numeric',
                'vascular_structures_numeric',
                'pigmentation_numeric',
                'streaks_numeric',
                'dots_and_globules_numeric',
                'regression_structures_numeric'
            ], batch_size=1, shuffle=False,
            map_fn='resize', one_hot_encoding=True, categories='auto', random_state=None, new_size=(256, 256),
            reshuffle_each_epoch=True,image_dir='/home/andreac/release_v0/images/'
    ):
        super(MultiTaskSequence, self).__init__()
        self.image_dir = image_dir
        self.df = df
        self._df = df.copy(deep=True)  # backup copy of original
        self.impath_col = impath_col
        self.labels_cols = labels_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.random_state = random_state
        self.new_size = new_size
        # pick map fn
        self.update_mapping_fn(map_fn)
        # encoding
        self.one_hot_encoding = one_hot_encoding
        self.categories = categories
        if self.one_hot_encoding:
            self.label_encoder = OneHotEncoder(sparse=False, dtype=np.float32, categories=categories)
        else:
            self.label_encoder = None
        # prepare data
        self.df, self.x, (self.y1, self.y2, self.y3, self.y4, self.y5, self.y6, self.y7) = self.prepare_data(
            self.df, self.shuffle, labels_cols=self.labels_cols, impath_col=self.impath_col
        )

    def prepare_data(self, df, shuffle=False, labels_cols=[
        'pigment_network_numeric',
        'blue_whitish_veil_numeric',
        'vascular_structures_numeric',
        'pigmentation_numeric',
        'streaks_numeric',
        'dots_and_globules_numeric',
        'regression_structures_numeric'
    ], impath_col='derm'):
        # shuffle data
        if shuffle:
            df = df.sample(frac=1.)

        # path loader
        x = df[impath_col]

        # encoding
        if self.one_hot_encoding:
            _y1 = np.reshape(df[labels_cols[0]].values, (-1, 1))
            y1 = self.label_encoder.fit_transform(_y1)
            _y2 = np.reshape(df[labels_cols[1]].values, (-1, 1))
            y2 = self.label_encoder.fit_transform(_y2)
            _y3 = np.reshape(df[labels_cols[2]].values, (-1, 1))
            y3 = self.label_encoder.fit_transform(_y3)
            _y4 = np.reshape(df[labels_cols[3]].values, (-1, 1))
            y4 = self.label_encoder.fit_transform(_y4)
            _y5 = np.reshape(df[labels_cols[4]].values, (-1, 1))
            y5 = self.label_encoder.fit_transform(_y5)
            _y6 = np.reshape(df[labels_cols[5]].values, (-1, 1))
            y6 = self.label_encoder.fit_transform(_y6)
            _y7 = np.reshape(df[labels_cols[6]].values, (-1, 1))
            y7 = self.label_encoder.fit_transform(_y7)

        else:
            y1 = df[labels_cols[0]].values
            y2 = df[labels_cols[1]].values
            y3 = df[labels_cols[2]].values
            y4 = df[labels_cols[3]].values
            y5 = df[labels_cols[4]].values
            y6 = df[labels_cols[5]].values
            y7 = df[labels_cols[6]].values

        return df, x, (y1, y2, y3, y4, y5, y6, y7)

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.x))
        batch_x = self.x[start_idx:end_idx]
        batch_y1 = self.y1[start_idx:end_idx]
        batch_y2 = self.y2[start_idx:end_idx]
        batch_y3 = self.y3[start_idx:end_idx]
        batch_y4 = self.y4[start_idx:end_idx]
        batch_y5 = self.y5[start_idx:end_idx]
        batch_y6 = self.y6[start_idx:end_idx]
        batch_y7 = self.y7[start_idx:end_idx]

        batch_x = tf.map_fn(self.map_function, batch_x, fn_output_signature=tf.float32)
        return (batch_x, (batch_y1, batch_y2, batch_y3, batch_y4, batch_y5, batch_y6, batch_y7))

    def on_epoch_end(self):
        if self.reshuffle_each_epoch:
            self.df, self.x, (self.y1, self.y2, self.y3, self.y4, self.y5, self.y6, self.y7) = self.prepare_data(
                self.df, self.shuffle, self.labels_cols)

    def update_mapping_fn(self, map_fn, updated_size=None):
        if updated_size is not None:
            print('changing resize size')
            self.new_size = updated_size

        print('updating mapping function')
        if isinstance(map_fn, str):
            if map_fn == 'resize':
                self.map_function = partial(load_resize, image_dir=self.image_dir, new_size=self.new_size)
            elif map_fn == 'aug':
                self.map_function = partial(load_aug_resize, image_dir=self.image_dir, new_size=self.new_size)
            elif map_fn == 'load':
                self.map_function = partial(load_images, image_dir=self.image_dir)
            else:
                raise Exception('no map function defined')
        else:
            self.map_function = map_fn

    def get_all_parameters(self):
        return {
            'derm_col': self.impath_col, 'labels_cols': self.labels_cols, 'batch_size': self.batch_size,
            'shuffle': self.shuffle, 'reshuffle_each_epoch': self.reshuffle_each_epoch,
            'random_state': self.random_state, 'one_hot_encoding': self.one_hot_encoding,
            'categories': self.categories, 'new_size': self.new_size
        }
