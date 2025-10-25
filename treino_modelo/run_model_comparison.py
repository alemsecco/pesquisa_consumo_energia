"""Prepare data, evaluate multiple regressors with TimeSeriesSplit and temporal holdout.
Saves:
 - treino_modelo/prepared_for_modeling.csv
 - treino_modelo/resultados_modelos_comparacao.csv
 - treino_modelo/modelo/best_model_comparison.joblib
"""
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_path = os.path.join(base, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
out_dir = os.path.join(base, 'treino_modelo')
model_dir = os.path.join(out_dir, 'modelo')
os.makedirs(model_dir, exist_ok=True)

print('Loading', csv_path)
df = pd.read_csv(csv_path, sep=';', encoding='utf-8', dtype=str)
print('Rows:', len(df))

# Clean Consumo similarly to comparacao.py
s = df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['Consumo_clean'] = pd.to_numeric(s, errors='coerce')
print('NaN targets after parse:', df['Consumo_clean'].isna().sum())

# create Ano, Mes from MesAno
if 'MesAno' in df.columns:
    df['MesAno_str'] = df['MesAno'].astype(str)
    df['Ano'] = df['MesAno_str'].str.slice(0,4).astype(int)
    df['Mes'] = df['MesAno_str'].str.slice(5,7).astype(int)
else:
    raise RuntimeError('MesAno not found')

# Select numerical weather columns by name patterns (keep original names)
num_cols_candidates = [c for c in df.columns if any(k in c for k in ['TEMPERATURA','PRECIPITACAO','PRESSAO','VENTO'])]
print('Numeric weather columns found:', num_cols_candidates)
# Clean numeric weather columns (they may be strings with ',' decimal and '.' thousand separators)
for col in num_cols_candidates:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"Cleaned {col}: NaNs={df[col].isna().sum()}")

# Generate groupwise lags based on Consumo_clean sorted by MesAno
# Need a time index; use MesAno_str as YYYY-MM-DD or YYYY-MM
# We'll sort by ['Regiao','Ano','Mes']
df_sorted = df.sort_values(['Regiao','Ano','Mes']).reset_index(drop=True)
# group and compute lags
for lag in [1,12]:
    df_sorted[f'lag{lag}'] = df_sorted.groupby('Regiao')['Consumo_clean'].shift(lag)
# rolling 3
df_sorted['roll3'] = df_sorted.groupby('Regiao')['Consumo_clean'].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# Winsorize target globally to reduce influence of extreme outliers
cons = df_sorted['Consumo_clean']
low, high = cons.dropna().quantile([0.005, 0.995])
print('Winsorize Consumo bounds:', low, high)
df_sorted['Consumo_wins'] = cons.clip(lower=low, upper=high)

# Decide features
features = ['Ano','Mes','lag1','lag12','roll3'] + num_cols_candidates + ['Regiao']
print('Using features:', features)

# Drop rows with NaN in required features or target
keep_mask = df_sorted['Consumo_wins'].notna()
for f in ['lag1','lag12','roll3']:
    keep_mask &= df_sorted[f].notna()

df_prepared = df_sorted.loc[keep_mask, :].copy().reset_index(drop=True)
print('Rows after dropping lags NaN:', len(df_prepared))

# Save prepared
prepared_path = os.path.join(out_dir, 'prepared_for_modeling.csv')
df_prepared.to_csv(prepared_path, index=False, sep=';', decimal=',')
print('Saved prepared to', prepared_path)

# Build X/y
X = df_prepared[features].copy()
y = df_prepared['Consumo_wins'].astype(float).copy()

# temporal holdout: last 20% of sorted df_prepared
n = len(X)
test_size = max(1, int(n * 0.20))
train_idx = list(range(0, n - test_size))
test_idx = list(range(n - test_size, n))
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
print('Train rows:', len(X_train), 'Test rows:', len(X_test))

# Preprocessor
numeric_features = [c for c in features if c != 'Regiao']
categorical_features = ['Regiao']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

# Candidate models
models = {}
# If tuned model exists, load and include
tuned_path = os.path.join(model_dir, 'rfr_tuned.joblib')
if os.path.exists(tuned_path):
    try:
        tuned = joblib.load(tuned_path)
        models['rfr_tuned'] = tuned
        print('Loaded rfr_tuned from', tuned_path)
    except Exception as e:
        print('Could not load tuned:', e)
# Standard RFR pipeline
models['rfr'] = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
# HistGB
models['histgb'] = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', HistGradientBoostingRegressor(random_state=42))])
# Huber
models['huber'] = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', HuberRegressor())])
# MLP
models['mlp'] = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))])

# Evaluate with TimeSeriesSplit on training set
tss = TimeSeriesSplit(n_splits=5)
results = []
for name, estimator in models.items():
    print('\nEvaluating', name)
    try:
        if isinstance(estimator, Pipeline):
            # cross_validate
            scores = cross_validate(estimator, X_train, y_train, cv=tss, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'], n_jobs=-1, error_score=np.nan)
            mae_cv = -scores['test_neg_mean_absolute_error'].mean()
            rmse_cv = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
            r2_cv = scores['test_r2'].mean()
        else:
            # assume already a fitted pipeline or estimator loaded
            # wrap with preprocessor manually
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', estimator)])
            scores = cross_validate(pipe, X_train, y_train, cv=tss, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'], n_jobs=-1, error_score=np.nan)
            mae_cv = -scores['test_neg_mean_absolute_error'].mean()
            rmse_cv = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
            r2_cv = scores['test_r2'].mean()
    except Exception as e:
        print('Cross-validation failed for', name, e)
        mae_cv = rmse_cv = r2_cv = np.nan

    # Fit on full train and evaluate on holdout
    try:
        if isinstance(estimator, Pipeline):
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
        else:
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', estimator)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
        mae_hold = mean_absolute_error(y_test, y_pred)
        rmse_hold = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_hold = r2_score(y_test, y_pred)
    except Exception as e:
        print('Holdout eval failed for', name, e)
        mae_hold = rmse_hold = r2_hold = np.nan

    print(f'{name} CV MAE {mae_cv:.2f} RMSE {rmse_cv:.2f} R2 {r2_cv:.4f} | Holdout MAE {mae_hold:.2f} RMSE {rmse_hold:.2f} R2 {r2_hold:.4f}')
    results.append({'model': name, 'cv_mae': mae_cv, 'cv_rmse': rmse_cv, 'cv_r2': r2_cv, 'hold_mae': mae_hold, 'hold_rmse': rmse_hold, 'hold_r2': r2_hold})

# Save results
res_df = pd.DataFrame(results).sort_values('hold_mae')
res_path = os.path.join(out_dir, 'resultados_modelos_comparacao.csv')
res_df.to_csv(res_path, index=False, sep=';', decimal=',')
print('\nSaved results to', res_path)

# ensure we pick the best non-null hold_mae
valid_rows = res_df[res_df['hold_mae'].notna()].copy()
if valid_rows.shape[0] == 0:
    print('Warning: no valid holdout results; saved results but will not save a best-model artifact')
else:
    best_row = valid_rows.iloc[0]
    best_name = best_row['model']
    print('Best model by holdout MAE:', best_name)
    best_est = models[best_name]
    try:
        if isinstance(best_est, Pipeline):
            best_est.fit(X, y)
            to_save = best_est
        else:
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', best_est)])
            pipe.fit(X, y)
            to_save = pipe
        save_path = os.path.join(model_dir, 'best_model_comparison.joblib')
        joblib.dump(to_save, save_path)
        print('Saved best model to', save_path)
    except Exception as e:
        print('Could not fit/save best model:', e)

print('Done.')
