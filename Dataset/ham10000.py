import os
import pandas as pd

meta_dir = os.path.join('..', '/home/andreac/HAM10000')
#image_dir = os.path.join('..', '/home/andreac/HAM10000/images')

df = pd.read_csv(os.path.join(meta_dir, 'HAM10000_metadata'))

def nev_or_mel(df):
    df = df.drop(df[df.dx == 'bkl'].index)
    df = df.drop(df[df.dx == 'bcc'].index)
    df = df.drop(df[df.dx == 'akiec'].index)
    df['dx'] = df['dx'].apply(lambda x: 1 if x == 'mel' else 0)
    return df
df = nev_or_mel(df)
df.rename(columns={'image_id' : 'derm', 'dx' : 'diagnosis_numeric'})
df = df[['image_id','dx']].sample(frac=1)
df.columns=['derm','diagnosis_numeric']
df['derm'] = df['derm'].astype(str) + '.jpg'
ham10000_df = df


