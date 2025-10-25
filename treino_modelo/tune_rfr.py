import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    from treino_modelo import treino
except Exception:
    # fallback: import by path
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location('treino', os.path.join(os.path.dirname(__file__), 'treino.py'))
    treino = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino)


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE, 'treino_modelo', 'modelo')
os.makedirs(MODEL_DIR, exist_ok=True)


def run_light_search(n_iter=12, n_splits=5, random_state=42):
    print('Carregando dados...')
    df = treino.load_data()
    print('Linhas:', len(df))

    print('Pré-processando...')
    X, y, pre = treino.preprocess(df)

    # build pipeline
    pipe = Pipeline([
        ('preprocessor', pre),
        ('regressor', RandomForestRegressor(random_state=random_state, n_jobs=-1))
    ])

    param_distributions = {
        'regressor__n_estimators': [100, 200, 400, 800],
        'regressor__max_depth': [None, 10, 20, 40, 80],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4, 10],
        'regressor__max_features': ['sqrt', 'log2', 0.2, 0.5, None],
        'regressor__bootstrap': [True, False]
    }

    tss = TimeSeriesSplit(n_splits=n_splits)

    print('Iniciando RandomizedSearchCV (n_iter=%d)...' % n_iter)
    rs = RandomizedSearchCV(pipe, param_distributions, n_iter=n_iter, scoring='neg_mean_absolute_error', cv=tss, n_jobs=-1, random_state=random_state, verbose=2)
    rs.fit(X, y)

    print('Busca concluída. Melhor score (neg MAE):', rs.best_score_)
    print('Melhores hiperparâmetros:', rs.best_params_)

    # temporal holdout: últimos 20% como teste (respeitando ordenação usada no preprocess)
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    best = rs.best_estimator_
    print('Reajustando melhor estimador no treino (0:80%) e avaliando no hold-out (80:100%)...')
    best.fit(X_train, y_train)
    preds = best.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f'Hold-out results -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}')

    out_path = os.path.join(MODEL_DIR, 'rfr_tuned.joblib')
    joblib.dump(best, out_path)
    print('Melhor pipeline salvo em', out_path)

    # Return a summary dict
    return {
        'best_params': rs.best_params_,
        'cv_neg_mae': rs.best_score_,
        'holdout_mae': mae,
        'holdout_rmse': rmse,
        'holdout_r2': r2,
        'model_path': out_path
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=12, help='Número de iterações do RandomizedSearchCV')
    parser.add_argument('--n_splits', type=int, default=5, help='Número de splits do TimeSeriesSplit')
    args = parser.parse_args()

    summary = run_light_search(n_iter=args.n_iter, n_splits=args.n_splits)
    print('\nResumo da busca:')
    for k, v in summary.items():
        print(f' - {k}: {v}')
