import os
try:
    import joblib
    def _load_model(p):
        return joblib.load(p)
except Exception:
    import pickle
    def _load_model(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from treino_modelo import treino
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location('treino', os.path.join(os.path.dirname(__file__), 'treino.py'))
    treino = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE, 'treino_modelo', 'modelo')

# prefer tuned global if present
candidates = ['rfr_global.joblib', 'rfr.joblib', 'rfr_tuned.joblib']
model_file = None
for c in candidates:
    p = os.path.join(MODEL_DIR, c)
    if os.path.exists(p):
        model_file = p
        break

if model_file is None:
    raise FileNotFoundError('Nenhum modelo encontrado em ' + MODEL_DIR)

print('Usando modelo:', model_file)
pipe = _load_model(model_file)

# load canonical data
print('Carregando dados...')
df = treino.load_data()
# ensure Consumo numeric
if 'Consumo' in df.columns:
    df['Consumo'] = df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df['Consumo'] = pd.to_numeric(df['Consumo'], errors='coerce')

# preprocess to get X,y and ensure lag features exist
X, y, pre = treino.preprocess(df)
# add MesAno_dt for ordering
df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')

regions = sorted(df['Regiao'].dropna().unique())
rows = []
for reg in regions:
    sub_idx = df.index[df['Regiao'] == reg]
    if len(sub_idx) == 0:
        continue
    sub = df.loc[sub_idx].copy()
    sub = sub.sort_values('MesAno_dt')
    order = sub.index
    n = len(order)
    if n < 12:
        rows.append({'Regiao': reg, 'n': n, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'notes': 'too_small'})
        continue
    split = int(n * 0.8)
    test_idx = order[split:]

    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    # predict
    preds = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    rows.append({'Regiao': reg, 'n': n, 'mae': mae, 'rmse': rmse, 'r2': r2, 'notes': ''})

out_df = pd.DataFrame(rows).sort_values('mae')
out_path = os.path.join(MODEL_DIR, '..', 'resultados_por_regiao_detailed.csv')
out_df.to_csv(out_path, index=False, sep=';', decimal=',')
print('Salvo em', out_path)
print(out_df.to_string())
