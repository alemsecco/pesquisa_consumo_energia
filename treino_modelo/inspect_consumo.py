import pandas as pd
import numpy as np
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
path = os.path.join(base, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
print('Reading', path)

df = pd.read_csv(path, sep=';', encoding='utf-8', dtype=str)
print('Total rows:', len(df))

s = df['Consumo'].astype(str)

# Raw stats
raw_nonnull = s.dropna()
print('\nRaw sample values (first 20):')
print(raw_nonnull.head(20).to_list())

# detect values containing unexpected characters or extremely long
def is_weird(v):
    if pd.isna(v):
        return False
    v = v.strip()
    # if there are letters
    if any(c.isalpha() for c in v):
        return True
    # if too many digits (excluding punctuation)
    digits = ''.join(ch for ch in v if ch.isdigit())
    if len(digits) > 12:
        return True
    return False

weird_mask = s.apply(is_weird)
print('\nWeird count:', weird_mask.sum())
print('Examples of weird values:')
print(s[weird_mask].head(20).to_list())

# Cleaning used in comparacao.py
s_clean1 = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
num1 = pd.to_numeric(s_clean1, errors='coerce')
print('\nAfter comparacao-style cleaning:')
print('NaN count:', num1.isna().sum())
print('Min/Max/median: ', num1.min(), num1.max(), num1.median())
print(num1.describe())

# Our stricter cleaning used in predict scripts: also drop absurd >1e12
num2 = num1.copy()
bad_large = num2.abs() > 1e12
print('\nCount > 1e12:', bad_large.sum())
print('Examples >1e12:')
print(num2[bad_large].head(10).to_list())

num2[bad_large] = np.nan
print('After setting >1e12 to NaN: NaN count:', num2.isna().sum())

# Compare rows kept
kept_comparacao = num1.notna().sum()
kept_strict = num2.notna().sum()
print(f'Kept after comparacao-clean: {kept_comparacao}, after strict-clean: {kept_strict} (dropped {kept_comparacao-kept_strict})')

# show rows where comparacao-clean yields a number but strict marks NaN
diff_idx = num1.notna() & num2.isna()
print('\nExamples where comparacao-clean kept but strict removed (first 10):')
print(df.loc[diff_idx, ['MesAno','Regiao','Consumo']].head(10).to_string())

# report top 10 largest values after comparacao clean
print('\nTop 10 largest values (after comparacao-clean):')
print(num1.sort_values(ascending=False).head(20).to_list())

# show distribution percentiles
print('\nPercentiles:')
print(num1.dropna().quantile([0.5,0.9,0.95,0.99,0.999]))
