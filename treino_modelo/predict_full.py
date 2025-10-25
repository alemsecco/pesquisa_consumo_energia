import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import joblib
    def _load_model(p):
        return joblib.load(p)
except Exception:
    import pickle
    def _load_model(p):
        with open(p, 'rb') as f:
            return pickle.load(f)

try:
    from treino_modelo import treino
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location('treino', os.path.join(os.path.dirname(__file__), 'treino.py'))
    treino = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE, 'treino_modelo', 'modelo')

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

# load data
print('Carregando dados...')
df = treino.load_data()
# ensure Consumo numeric
if 'Consumo' in df.columns:
    df['Consumo'] = df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df['Consumo'] = pd.to_numeric(df['Consumo'], errors='coerce')

# preprocess to get X,y
X, y, pre = treino.preprocess(df)
# time order
if 'MesAno' in df.columns:
    df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')
    order = df.sort_values('MesAno_dt').index
else:
    order = df.index

n = len(order)
split = int(n * 0.8)
train_idx = order[:split]
test_idx = order[split:]

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

print('Gerando previsões para holdout (últimos', n - split, 'linhas)...')
preds = pipe.predict(X_test)

mae = float(mean_absolute_error(y_test, preds))
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
r2 = float(r2_score(y_test, preds))

out = df.loc[test_idx].copy()
out['Pred_Consumo'] = preds
out_path = os.path.join(MODEL_DIR, 'predictions_full_' + os.path.basename(model_file).replace('.joblib','') + '.csv')
out.to_csv(out_path, index=False, sep=';', decimal=',')

print('Metrics on holdout:')
print('MAE:', mae)
print('RMSE:', rmse)
print('R2:', r2)
print('Predictions saved to', out_path)
print('\nSample predictions:')
print(out[['MesAno','Regiao','Consumo','Pred_Consumo']].head(10).to_string())
