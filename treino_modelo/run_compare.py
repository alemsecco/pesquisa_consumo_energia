import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from treino_modelo.treino import load_data, add_lag_features, time_series_compare, DATA_PATH
import pandas as pd

# Carrega e cria lags
print('Carregando dados e criando lags...')
df = load_data(DATA_PATH)
df_lag = add_lag_features(df)
# Mostrar primeiras linhas com lags
cols = ['MesAno','Regiao','Consumo','lag1','lag12','roll3']
print('\nAmostra com lags (primeiras 10 linhas):')
print(df_lag[cols].head(10).to_string())

# Rodar comparação temporal
print('\nExecutando comparação temporal (TimeSeriesSplit)...')
results, best = time_series_compare(df, n_splits=5)
print('\nResultados por modelo:')
for m, r in results.items():
    print(f"- {m}: MAE={r['mae_mean']:.3f}, RMSE={r['rmse_mean']:.3f}, R2={r['r2_mean']:.3f}")
print('\nMelhor modelo (por MAE):', best)
