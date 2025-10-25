import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
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
os.makedirs(MODEL_DIR, exist_ok=True)


def load_population_by_region():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pop_dir = os.path.join(base, 'dados', 'população')
    pop_by_region = {}
    # region files named like 'Censo 2022 - Crescimento Populacional - Nordeste.csv'
    for fn in os.listdir(pop_dir):
        if fn.lower().startswith('censo') and fn.lower().endswith('.csv'):
            path = os.path.join(pop_dir, fn)
            try:
                dfp = pd.read_csv(path, sep=';', encoding='utf-8')
                # try to find row for 2022
                if 'Ano da pesquisa' in dfp.columns and 'População(pessoas)' in dfp.columns:
                    # coerce population to numeric (handles thousands separators)
                    col = dfp['População(pessoas)'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    col_num = pd.to_numeric(col, errors='coerce')
                    if 2022 in dfp['Ano da pesquisa'].values:
                        pop2022 = int(col_num.loc[dfp['Ano da pesquisa'] == 2022].dropna().iloc[0])
                    else:
                        # fallback to last non-null population
                        pop2022 = int(col_num.dropna().iloc[-1])
                    # extract region name from filename
                    parts = fn.split('-')
                    region = parts[-1].replace('.csv', '').strip()
                    pop_by_region[region] = pop2022
            except Exception:
                continue
    return pop_by_region


def add_population_features(df):
    df = df.copy()
    pop_map = load_population_by_region()
    # add Pop_2022 if map has region
    df['Pop_2022'] = df['Regiao'].map(pop_map).astype(float)
    # If population missing, fill with median
    df['Pop_2022'] = df['Pop_2022'].fillna(np.nanmedian(list(pop_map.values())) if pop_map else df['Pop_2022'].median())
    # per-capita feature (per 1000 inhabitants)
    df['Consumo_per_1000hab'] = df['Consumo'] / (df['Pop_2022'] / 1000.0)
    return df


def train_global_and_regions():
    print('Carregando dados...')
    df = treino.load_data()
    print('Linhas:', len(df))

    # ensure Consumo numeric (robust: only clean strings, keep numeric types as-is)
    def _clean_series(s):
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors='coerce')
        # assume strings with '.' thousands and ',' decimal
        return pd.to_numeric(s.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

    df['Consumo'] = _clean_series(df['Consumo'])

    # add population features
    df2 = add_population_features(df)

    # Global model: use tuned pipeline if present
    tuned_path = os.path.join(MODEL_DIR, 'rfr_tuned.joblib')
    if os.path.exists(tuned_path):
        print('Carregando pipeline ajustada existente:', tuned_path)
        best_pipe = joblib.load(tuned_path)
    else:
        print('Modelo ajustado não encontrado; treine rfr_tuned primeiro')
        best_pipe = None

    # Prepare X,y with new features by calling preprocess on augmented df
    X, y, pre = treino.preprocess(df2)

    # Train global model: fit best_pipe on first 80% (time-ordered) and save
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if best_pipe is not None:
        print('Construindo pipeline final usando o preprocessor atual e hiperparâmetros ajustados...')
        # extract regressor params and build a new pipeline that fits the current preprocessor
        try:
            tuned_reg = best_pipe.named_steps.get('regressor', None)
            tuned_params = tuned_reg.get_params() if tuned_reg is not None else None
        except Exception:
            tuned_params = None

        if tuned_params:
            rf = RandomForestRegressor(**{k: v for k, v in tuned_params.items() if k in RandomForestRegressor().get_params()})
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        from sklearn.pipeline import Pipeline
        final_pipe = Pipeline([('preprocessor', pre), ('regressor', rf)])
        final_pipe.fit(X_train, y_train)
        preds = final_pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)
        out_global = os.path.join(MODEL_DIR, 'rfr_global.joblib')
        joblib.dump(final_pipe, out_global)
        print('Global model saved to', out_global)
        print('Global holdout MAE:', mae, 'RMSE:', rmse, 'R2:', r2)
    else:
        print('No tuned pipeline to use for global training.')

    # Per-region training: train a model per Regiao using same best hyperparameters if available
    regions = df2['Regiao'].dropna().unique()
    rows = []
    # extract regressor params from tuned pipeline if possible
    reg_params = None
    if best_pipe is not None:
        try:
            reg = best_pipe.named_steps['regressor']
            reg_params = reg.get_params()
        except Exception:
            reg_params = None

    for reg in sorted(regions):
        sub = df2[df2['Regiao'] == reg].copy()
        if sub.shape[0] < 12:
            print('Regiao', reg, 'tem poucas observações, pulando')
            continue
        print('Treinando por região:', reg)
        Xr, yr, prer = treino.preprocess(sub)
        # time-order by MesAno
        sub['MesAno_dt'] = pd.to_datetime(sub['MesAno'] + '-01', errors='coerce')
        order = sub.sort_values('MesAno_dt').index
        Xr = Xr.loc[order]
        yr = yr.loc[order]

        n_r = len(Xr)
        split_r = int(n_r * 0.8)
        Xr_train, Xr_test = Xr.iloc[:split_r], Xr.iloc[split_r:]
        yr_train, yr_test = yr.iloc[:split_r], yr.iloc[split_r:]

        if reg_params:
            model = RandomForestRegressor(**{k: v for k, v in reg_params.items() if k in RandomForestRegressor().get_params()})
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        from sklearn.pipeline import Pipeline
        pipe_reg = Pipeline([('preprocessor', prer), ('regressor', model)])
        pipe_reg.fit(Xr_train, yr_train)
        preds_r = pipe_reg.predict(Xr_test)
        mae_r = mean_absolute_error(yr_test, preds_r)
        rmse_r = float(np.sqrt(mean_squared_error(yr_test, preds_r)))
        r2_r = r2_score(yr_test, preds_r)
        model_path = os.path.join(MODEL_DIR, f'rfr_region_{reg.replace(" ","_")}.joblib')
        joblib.dump(pipe_reg, model_path)
        rows.append({'Regiao': reg, 'n': n_r, 'mae': float(mae_r), 'rmse': float(rmse_r), 'r2': float(r2_r), 'model_path': model_path})
        print('Saved region model for', reg, '->', model_path)

    out_df = pd.DataFrame(rows).sort_values('mae')
    out_csv = os.path.join(MODEL_DIR, '..', 'resultados_por_regiao_models.csv')
    out_df.to_csv(out_csv, index=False, sep=';', decimal=',')
    print('Per-region results saved to', out_csv)
    return out_df


if __name__ == '__main__':
    train_global_and_regions()
