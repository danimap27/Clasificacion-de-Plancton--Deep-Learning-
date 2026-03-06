import pandas as pd
from PIL import Image
import os

df = pd.read_csv('datos/datos/entrenamiento/entrenamiento.csv', sep=';')
print('=== CLASSES ===')
print('Total classes:', df['Clase'].nunique())
print(df['Clase'].value_counts())

print('\n=== GROUPS ===')
print('Total groups:', df['GrupoFuncional'].nunique())
print(df.groupby(['GrupoFuncional', 'Clase']).size())

dims = []
base = 'datos/datos/entrenamiento'
for imp in df['Imagen'].head(1000).tolist():
    img = Image.open(os.path.join(base, imp))
    dims.append(img.size)

ds = pd.DataFrame(dims, columns=['width', 'height'])
print('\n=== DIMENSIONS (First 1000) ===')
print(ds.describe())
