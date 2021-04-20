import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

meta_dir = os.path.join('..', '/home/andreac/release_v0/meta')
image_dir = os.path.join('..', '/home/andreac/release_v0/images')

df = pd.read_csv(os.path.join(meta_dir, 'dataframe.csv'))
df['diagnosis_numeric']=df['diagnosis_numeric'].apply(lambda x: 1 if x == 2 else 0) #MEL or NOT-MEL 'mapping'

train_indexes = list(pd.read_csv(os.path.join(meta_dir, 'train_indexes.csv'))['indexes'])
valid_indexes = list(pd.read_csv(os.path.join(meta_dir, 'valid_indexes.csv'))['indexes'])
test_indexes = list(pd.read_csv(os.path.join(meta_dir, 'test_indexes.csv'))['indexes'])

train_df = df.iloc[train_indexes]
valid_df = df.iloc[valid_indexes]
test_df = df.iloc[test_indexes]