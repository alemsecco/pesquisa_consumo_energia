import pandas as pd
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
consumo_path = os.path.join(base, 'consumo_energia', 'Dados_residencial.csv')
temp_path = os.path.join(base, 'temperatura', 'temperatura_mensal_com_regiao_final.csv')

print('Reading consumo:', consumo_path)
dfc = pd.read_csv(consumo_path, sep=';', decimal=',', encoding='utf-8')
print('Consumo columns:', dfc.columns.tolist())
print('\nConsumo head:\n', dfc.head(8).to_string())
print('\nConsumo dtypes:\n', dfc.dtypes)
if 'Consumo' in dfc.columns:
    print('\nConsumo sample unique values (first 20):')
    print(dfc['Consumo'].astype(str).unique()[:20])
    print('Consumo non-null count:', dfc['Consumo'].notna().sum())
    try:
        s = pd.to_numeric(dfc['Consumo'], errors='coerce')
        print('Consumo sum (numeric, ignoring NaN):', s.sum())
    except Exception as e:
        print('Error converting Consumo to numeric:', e)

print('\n---\nReading temperatura:', temp_path)
dfT = pd.read_csv(temp_path, sep=';', decimal=',', encoding='utf-8')
print('Temp columns:', dfT.columns.tolist())
print('\nTemp head:\n', dfT.head(8).to_string())
print('\nTemp dtypes:\n', dfT.dtypes)

meas = [
    'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
    'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
    'PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)',
    'TEMPERATURA MEDIA, MENSAL (AUT)(°C)',
    'VENTO, VELOCIDADE MAXIMA MENSAL (AUT)(m/s)',
    'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)'
]

meas_present = [c for c in meas if c in dfT.columns]
print('\nMeasurement columns present:', meas_present)
for c in meas_present:
    print(f"{c} non-null count:", dfT[c].notna().sum(), ' — sample values:', dfT[c].dropna().unique()[:5])

print('\nUnique Regiao in consumo (first 30):')
if 'Regiao' in dfc.columns:
    print(sorted(dfc['Regiao'].dropna().unique())[:30])
else:
    print('Regiao not in consumo')

print('\nUnique Regiao in temp (first 30):')
if 'Regiao' in dfT.columns:
    regT = sorted(dfT['Regiao'].dropna().unique())
    print(regT[:30])
else:
    print('Regiao not in temp')

if 'Regiao' in dfc.columns and 'Regiao' in dfT.columns:
    regC = set(dfc['Regiao'].dropna().unique())
    regTset = set(dfT['Regiao'].dropna().unique())
    print('\nRegions intersection count:', len(regC & regTset))
    print('Regions only in consumo:', sorted(regC - regTset))
    print('Regions only in temp (sample):', sorted(list(regTset - regC))[:20])

print('\nRows in temp where TEMPERATURA not null (sample 10):')
if 'TEMPERATURA MEDIA, MENSAL (AUT)(°C)' in dfT.columns:
    print(dfT[dfT['TEMPERATURA MEDIA, MENSAL (AUT)(°C)'].notna()].head(10).to_string())
else:
    print('TEMPERATURA column not present by that exact name')

print('\nInspection done')
