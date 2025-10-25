import os
import sys
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

if len(sys.argv) < 2:
    print('Usage: python predict_with_model.py <model_filename>')
    sys.exit(1)

model_name = sys.argv[1]
model_path = os.path.join(MODEL_DIR, model_name)
if not os.path.exists(model_path):
    raise FileNotFoundError(model_path)

print('Using model:', model_path)
pipe = _load_model(model_path)

# load data
print('Loading data...')
df = treino.load_data()
# robust Consumo cleaning
if 'Consumo' in df.columns:
    s = df['Consumo'].astype(str)
    s = s.str.replace('\xa0', '', regex=False)
    s = s.str.replace(' ', '', regex=False)
    s = s.str.replace('.', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    # coerce to numeric, large values > 1e12 treated as NaN
    df['Consumo'] = pd.to_numeric(s, errors='coerce')
    df.loc[df['Consumo'].abs() > 1e12, 'Consumo'] = np.nan

# Clean numeric weather columns so imputers receive numeric dtype
num_cols_candidates = [c for c in df.columns if any(k in c for k in ['TEMPERATURA','PRECIPITACAO','PRESSAO','VENTO'])]
for col in num_cols_candidates:
    df[col] = df[col].astype(str).str.replace('\xa0', '', regex=False)
    df[col] = df[col].str.replace(' ', '', regex=False)
    df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # leave NaNs as-is; preprocessor will impute
    # print sample of NaNs count
    print(f'Cleaned {col}: NaNs={df[col].isna().sum()}')

# preprocess to get X,y
# Build lags and date-derived cols on a working copy so we can provide exactly
# the features the pipeline expects at predict time.
df_work = df.copy()
# add lag features (function sorts/groupby internally)
df_work = treino.add_lag_features(df_work)
# ensure Ano/Mes
if 'MesAno' in df_work.columns:
    df_work['MesAno'] = df_work['MesAno'].astype(str)
    df_work['Ano'] = df_work['MesAno'].str.slice(0,4).astype(int)
    df_work['Mes'] = df_work['MesAno'].str.slice(5,7).astype(int)

# Determine time ordering and holdout split
if 'MesAno' in df_work.columns:
    df_work['MesAno_dt'] = pd.to_datetime(df_work['MesAno'] + '-01', errors='coerce')
    order = df_work.sort_values('MesAno_dt').index
else:
    order = df_work.index

n = len(order)
split = int(n * 0.8)
train_idx = order[:split]
test_idx = order[split:]

# Determine required columns for the model
required_cols = None
if hasattr(pipe, 'feature_names_in_'):
    required_cols = list(pipe.feature_names_in_)
else:
    # try to infer from a ColumnTransformer inside the pipeline
    required_cols = []
    try:
        pre = pipe.named_steps.get('preprocessor') if isinstance(pipe, (type,)) is False else None
        pre = getattr(pipe, 'named_steps', {}).get('preprocessor', None)
        if pre is not None and hasattr(pre, 'transformers_'):
            # transformers_[i] = (name, transformer, columns)
            for tr in pre.transformers_:
                cols = tr[2]
                if isinstance(cols, (list, tuple)):
                    required_cols.extend(list(cols))
    except Exception:
        required_cols = []

if not required_cols:
    # fallback: use the same features as treino.preprocess would produce
    X_tmp, y_tmp, _ = treino.preprocess(df_work)
    required_cols = X_tmp.columns.tolist()

print('Model requires columns:', required_cols)

# Build X_test with required columns (pull from df_work where possible)
X_test = pd.DataFrame(index=test_idx)
for c in required_cols:
    if c in df_work.columns:
        X_test[c] = df_work.loc[test_idx, c]
    else:
        # create missing columns filled with NaN (imputer in pipeline will handle)
        X_test[c] = np.nan

# target (cleaned Consumo) from df_work
y_test = pd.to_numeric(df_work.loc[test_idx, 'Consumo'], errors='coerce')

print('Predicting on holdout (size', len(test_idx), ')...')
# drop rows where y_test is NaN (bad/malformed targets)
mask = ~y_test.isna()
if mask.sum() == 0:
    print('No valid targets in holdout after cleaning; aborting')
    sys.exit(1)
X_test = X_test.loc[mask]
y_test = y_test.loc[mask]

preds = pipe.predict(X_test)

mae = float(mean_absolute_error(y_test, preds))
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
r2 = float(r2_score(y_test, preds))

out = df.loc[test_idx].copy()
out['Pred_Consumo'] = np.nan
out.loc[X_test.index, 'Pred_Consumo'] = preds
out_file = os.path.join(MODEL_DIR, f'predictions_full_{os.path.splitext(model_name)[0]}.csv')
out.to_csv(out_file, index=False, sep=';', decimal=',')

print('Model:', model_name)
print('MAE:', mae)
print('RMSE:', rmse)
print('R2:', r2)
print('Saved predictions to', out_file)
print('\nSample:')
print(out[['MesAno','Regiao','Consumo','Pred_Consumo']].head(8).to_string())
