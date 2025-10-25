import os
import pandas as pd

base = r'C:/Users/alems/Documents/pesquisa/energia_consumo'
pop_dir = os.path.join(base, 'dados', 'população')
print('Files in', pop_dir)
for fn in sorted(os.listdir(pop_dir)):
    path = os.path.join(pop_dir, fn)
    print('\n---', fn)
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8')
        print('columns:', list(df.columns))
        print(df.head(3).to_string())
    except Exception as e:
        print('Failed to read', fn, e)
