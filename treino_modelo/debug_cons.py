import os
import pandas as pd

base = r'C:/Users/alems/Documents/pesquisa/energia_consumo'
path = os.path.join(base, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
for i in range(min(10, len(df))):
    raw = df['Consumo'].iloc[i]
    cleaned = str(raw).replace('.', '').replace(',', '.')
    try:
        num = pd.to_numeric(cleaned)
    except Exception as e:
        num = e
    print(i, 'raw->', raw, 'cleaned->', cleaned, 'num->', num)
