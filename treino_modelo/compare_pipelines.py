import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from treino_modelo import treino
    from treino_modelo import pipelines
except Exception:
    import importlib.util, os as _os
    spec = importlib.util.spec_from_file_location('treino', _os.path.join(_os.path.dirname(__file__), 'treino.py'))
    treino = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino)
    spec2 = importlib.util.spec_from_file_location('pipelines', _os.path.join(_os.path.dirname(__file__), 'pipelines.py'))
    pipelines = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pipelines)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(BASE, 'analysis')
os.makedirs(OUT, exist_ok=True)


def evaluate_pipelines(n_splits=5):
    print('Carregando dados...')
    df = treino.load_data()
    X, y, pre = treino.preprocess(df)

    # Build pipelines
    tree_pipe = pipelines.build_tree_pipeline(pre)
    sens_pipe_std = pipelines.build_sensitive_pipeline(pre, scaler='standard', use_power=True)
    sens_pipe_rob = pipelines.build_sensitive_pipeline(pre, scaler='robust', use_power=True)

    pipes = {'tree': tree_pipe, 'sensitive_std': sens_pipe_std, 'sensitive_robust': sens_pipe_rob}

    tss = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for name, pipe in pipes.items():
        maes, rmses, r2s = [], [], []
        print('Evaluando pipeline:', name)
        for train_idx, test_idx in tss.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # fit and predict
            try:
                pipe.fit(X_train, y_train)
            except Exception as e:
                print('Error fitting', name, e)
                continue
            preds = pipe.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            r2s.append(r2_score(y_test, preds))

        records.append({'pipeline': name, 'mae_mean': float(np.mean(maes)) if maes else np.nan, 'rmse_mean': float(np.mean(rmses)) if rmses else np.nan, 'r2_mean': float(np.mean(r2s)) if r2s else np.nan})

    out = pd.DataFrame(records).sort_values('mae_mean')
    out.to_csv(os.path.join(OUT, 'pipeline_comparison.csv'), index=False, sep=';', decimal=',')
    print('Saved pipeline_comparison.csv')
    print(out.to_string(index=False))
    return out


if __name__ == '__main__':
    evaluate_pipelines()
