import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from Utils.utils import balance_df

meta_dir = os.path.join('..', '/home/andreac/release_v0/meta')
image_dir = os.path.join('..', '/home/andreac/release_v0/images')

df = pd.read_csv(os.path.join(meta_dir, 'dataframe.csv'))

train_indexes = list(pd.read_csv(os.path.join(meta_dir, 'train_indexes.csv'))['indexes'])
valid_indexes = list(pd.read_csv(os.path.join(meta_dir, 'valid_indexes.csv'))['indexes'])
test_indexes = list(pd.read_csv(os.path.join(meta_dir, 'test_indexes.csv'))['indexes'])

train_df = df.iloc[train_indexes]
valid_df = df.iloc[valid_indexes]
test_df = df.iloc[test_indexes]

#MEL && NEV cases mapping
def nev_or_mel(df):
    df = df.drop(df[df.diagnosis_numeric==0].index)
    df = df.drop(df[df.diagnosis_numeric==3].index)
    df = df.drop(df[df.diagnosis_numeric==4].index)
    df['diagnosis_numeric']=df['diagnosis_numeric'].apply(lambda x: 1 if x == 2 else 0)
    return df
train_df = nev_or_mel(train_df)
valid_df = nev_or_mel(valid_df)
test_df = nev_or_mel(test_df)

balance_train_df = balance_df(train_df)
balance_valid_df = balance_df(valid_df)
balance_test_df = balance_df(test_df)



