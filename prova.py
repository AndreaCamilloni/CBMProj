import os
import pandas as pd

meta_dir = os.path.join('..', '/Users/andre/Desktop/Internship BII/release_v0/meta')
#prova=os.path.join(base_skin_dir, '*', '*.jpg')
#glob_prova = glob(prova)
#print(glob_prova)
image_dir = os.path.join('..', '/Users/andre/Desktop/Internship BII/release_v0/images')

df = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

df['path'] = df['derm']
df['diagnosis_idx']=pd.Categorical(df['diagnosis']).codes
df = df.loc[:,['case_num','derm', 'diagnosis', 'diagnosis_idx','path']]
df.head()