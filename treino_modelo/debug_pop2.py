import os
import pandas as pd
import numpy as np

base = r'C:/Users/alems/Documents/pesquisa/energia_consumo'
pop_dir = os.path.join(base, 'dados', 'população')
pop_map = {}
for fn in os.listdir(pop_dir):
    if fn.lower().startswith('censo') and fn.lower().endswith('.csv'):
        path = os.path.join(pop_dir, fn)
        dfp = pd.read_csv(path, sep=';', encoding='utf-8')
        if 'Ano da pesquisa' in dfp.columns and 'População(pessoas)' in dfp.columns:
            col = dfp['População(pessoas)'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            col_num = pd.to_numeric(col, errors='coerce')
            if 2022 in dfp['Ano da pesquisa'].values:
                pop2022 = int(col_num.loc[dfp['Ano da pesquisa'] == 2022].dropna().iloc[0])
            else:
                pop2022 = int(col_num.dropna().iloc[-1])
            region = fn.split('-')[-1].replace('.csv','').strip()
            pop_map[region] = pop2022

print('pop_map:', pop_map)

path = os.path.join(base, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
df['Consumo'] = pd.to_numeric(df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
df['Pop_2022'] = df['Regiao'].map(pop_map).astype(float)
df['Pop_2022'] = df['Pop_2022'].fillna(np.nanmedian(list(pop_map.values())) if pop_map else df['Pop_2022'].median())
df['Consumo_per_1000hab'] = df['Consumo'] / (df['Pop_2022'] / 1000.0)
print(df[['Regiao','Pop_2022','Consumo','Consumo_per_1000hab']].head(10).to_string())
