import matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob #The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell - results are returned in arbitrary order
import math
import tensorflow as tf
from pasta.augment import inline
from tensorflow.keras import utils
from tensorflow import keras
from keras.utils import Sequence
from tensorflow.keras import *
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Add, Dense

#For splitting in test and validation sets
from sklearn.model_selection import train_test_split

#For use of Keras Sequence class that allows the load of dataset in batches
from skimage.io import imread
from skimage.transform import resize

#For some image plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

meta_dir = os.path.join('..', '/Users/andre/Desktop/Internship BII/release_v0/meta')
#prova=os.path.join(base_skin_dir, '*', '*.jpg')
#glob_prova = glob(prova)
#print(glob_prova)
image_dir = os.path.join('..', '/Users/andre/Desktop/Internship BII/release_v0/images')

df = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

df['path'] = df['derm']
df['diagnosis_idx']=pd.Categorical(df['diagnosis']).codes
df = df.loc[:,['case_num','derm', 'diagnosis', 'diagnosis_idx','path']]

mtl_df = df.loc[:, ['case_num', 'diagnosis_idx','derm']]

for i in range(7):
    mtl_df.insert(i, str(i), 0)

def task_column(row):
    task_class = str(row.diagnosis_idx)
    row[task_class] = 1
    return row

#For avoid truncation of path string in output
pd.set_option('display.max_colwidth', None)
mtl_df_t = mtl_df.apply(task_column, axis='columns')
mtl_df_t =mtl_df_t.loc[:, ['0','1','2','3','4','5','6','derm']]