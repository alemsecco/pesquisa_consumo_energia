import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
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
MODEL_PATH = os.path.join(MODEL_DIR, 'rfr_tuned.joblib')
OUT_CSV = os.path.join(BASE, 'treino_modelo', 'resultados_por_regiao.csv')


def evaluate_model_by_region(model_path=MODEL_PATH, n_splits=5):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    print('Carregando dados e modelo...')
    df = treino.load_data()
    X, y, pre = treino.preprocess(df)
    model = joblib.load(model_path)

    # reattach MesAno_dt and Regiao to X for grouping
    df_full = df.copy()
    df_full['MesAno_dt'] = pd.to_datetime(df_full['MesAno'] + '-01', errors='coerce')

    regions = df_full['Regiao'].dropna().unique()
    rows = []

    for reg in sorted(regions):
        mask = df_full['Regiao'] == reg
        sub = df_full.loc[mask].copy()
        n = len(sub)
        if n == 0:
            continue
        sub = sub.sort_values('MesAno_dt')

        # build X_region and y_region as in preprocess
        X_reg = X.loc[sub.index]
        y_reg = y.loc[sub.index]

        if len(X_reg) < 3:
            # too few points; skip
            print(f'Regiao {reg}: poucos dados ({len(X_reg)} linhas), pulando')
            continue

        maes = []
        rmses = []
        r2s = []

        if len(X_reg) > n_splits:
            tss = TimeSeriesSplit(n_splits=n_splits)
            for train_idx, test_idx in tss.split(X_reg):
                X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
                y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
                # fit on train and predict test using the pipeline
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                except Exception:
                    # if model is already fitted pipeline, just predict (we'll avoid refit)
                    preds = model.predict(X_test)

                maes.append(mean_absolute_error(y_test, preds))
                rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
                r2s.append(r2_score(y_test, preds))
        else:
            # fallback hold-out 80/20
            split = int(len(X_reg) * 0.8)
            if split < 1:
                print(f'Regiao {reg}: série muito curta para hold-out, pulando')
                continue
            X_train, X_test = X_reg.iloc[:split], X_reg.iloc[split:]
            y_train, y_test = y_reg.iloc[:split], y_reg.iloc[split:]
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception:
                preds = model.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            r2s.append(r2_score(y_test, preds))

        rows.append({'Regiao': reg, 'n': n, 'mae_mean': float(np.mean(maes)), 'rmse_mean': float(np.mean(rmses)), 'r2_mean': float(np.mean(r2s))})

    df_out = pd.DataFrame(rows).sort_values('mae_mean')
    df_out.to_csv(OUT_CSV, index=False, sep=';', decimal=',')
    print('Resultados por região salvos em', OUT_CSV)
    print(df_out.head(10).to_string(index=False))
    return df_out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Caminho para o modelo salvo')
    parser.add_argument('--n_splits', type=int, default=5, help='Número de splits do TimeSeriesSplit por região')
    args = parser.parse_args()

    evaluate_model_by_region(model_path=args.model, n_splits=args.n_splits)
