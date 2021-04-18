import os
import pandas as pd

from Dataset import df

df['diagnosis_numeric']=df['diagnosis_numeric'].apply(lambda x: 1 if x == 2 else 0) #MEL or NOT-MEL 'mapping'

