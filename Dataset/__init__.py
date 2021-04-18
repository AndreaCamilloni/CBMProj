import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

meta_dir = os.path.join('..', '/home/andreac/release_v0/meta')
image_dir = os.path.join('..', '/home/andreac/release_v0/images')

df = pd.read_csv(os.path.join(meta_dir, 'dataframe.csv'))

train_indexes = list(pd.read_csv(os.path.join(meta_dir, 'train_indexes.csv'))['indexes'])
valid_indexes = list(pd.read_csv(os.path.join(meta_dir, 'valid_indexes.csv'))['indexes'])
test_indexes = list(pd.read_csv(os.path.join(meta_dir, 'test_indexes.csv'))['indexes'])



