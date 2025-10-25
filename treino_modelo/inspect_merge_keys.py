import pandas as pd
import os

base = r'C:/Users/alems/Documents/pesquisa/energia_consumo'
path = os.path.join(base, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
print('consumo file columns:', list(df.columns))
print(df[['MesAno','Regiao']].head(5).to_string())

popath = os.path.join(base, 'dados', 'população', 'POP2022_Brasil_e_UFs.csv')
if os.path.exists(popath):
    pop = pd.read_csv(popath, sep=';', encoding='utf-8')
    print('\nPOP file columns:', list(pop.columns))
    print(pop.head(5).to_string())
else:
    print('POP file not found at', popath)
