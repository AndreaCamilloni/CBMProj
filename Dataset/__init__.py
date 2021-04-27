import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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
def binary_diagnosis(df):
    df['diagnosis_numeric']=df['diagnosis_numeric'].apply(lambda x: 10 if ((x == 4) or (x == 3) or (x == 0)) else x)
    delete_row = df[df['diagnosis_numeric']==10].index
    df = df.drop(delete_row)
    df['diagnosis_numeric']=df['diagnosis_numeric'].apply(lambda x: 1 if x == 2 else 0)
    return df

train_df = binary_diagnosis(train_df)
valid_df = binary_diagnosis(valid_df)
test_df = binary_diagnosis(test_df)
