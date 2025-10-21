import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# prefer the file in dados/manipulação, fallback to dados/
p1 = os.path.join(BASE, 'dados', 'manipulação', 'consumo_vs_temperatura_por_regiao.csv')
p2 = os.path.join(BASE, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
DATA_PATH = p1 if os.path.exists(p1) else p2
MODEL_DIR = os.path.join(BASE, 'treino_modelo', 'modelo')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
    return df


def add_lag_features(df: pd.DataFrame):
    # expects df with MesAno and Regiao and Consumo numeric
    df = df.copy()
    df['Consumo'] = pd.to_numeric(df['Consumo'], errors='coerce')
    # ensure MesAno is datetime-like for sorting
    df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')
    df = df.sort_values(['Regiao', 'MesAno_dt'])
    df['lag1'] = df.groupby('Regiao')['Consumo'].shift(1)
    df['lag12'] = df.groupby('Regiao')['Consumo'].shift(12)
    df['roll3'] = df.groupby('Regiao')['Consumo'].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def preprocess(df: pd.DataFrame):
    # Drop rows without MesAno or Regiao
    df = df.dropna(subset=['MesAno', 'Regiao']).copy()
    # add lag features
    df = add_lag_features(df)

    # Parse MesAno to extract year and month
    df['MesAno'] = df['MesAno'].astype(str)
    df['Ano'] = df['MesAno'].str.slice(0,4).astype(int)
    df['Mes'] = df['MesAno'].str.slice(5,7).astype(int)

    # Target: Consumo
    df['Consumo'] = pd.to_numeric(df['Consumo'], errors='coerce')

    # Feature columns: Ano, Mes, Regiao (one-hot), TEMPERATURA MEDIA, PRECIPITACAO, PRESSAO, VENTO
    features = []
    features += ['Ano', 'Mes']
    # include lags if present
    for lf in ['lag1', 'lag12', 'roll3']:
        if lf in df.columns:
            features.append(lf)
    if 'TEMPERATURA MEDIA, MENSAL (AUT)(°C)' in df.columns:
        features.append('TEMPERATURA MEDIA, MENSAL (AUT)(°C)')
    if 'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)' in df.columns:
        features.append('PRECIPITACAO TOTAL, MENSAL (AUT)(mm)')
    if 'PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)' in df.columns:
        features.append('PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)')
    if 'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)' in df.columns:
        features.append('VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)')

    X = df[features + ['Regiao']].copy()
    y = df['Consumo'].copy()

    # Column transformer: impute numeric, one-hot encode Regiao
    numeric_features = [c for c in features if c not in ['Regiao']]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_features = ['Regiao']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor


def time_series_compare(df: pd.DataFrame, n_splits=5):
    # Prepare dataset
    X, y, pre = preprocess(df)
    # convert MesAno to datetime index for ordering
    df_idx = df.copy()
    df_idx['MesAno_dt'] = pd.to_datetime(df_idx['MesAno'] + '-01', errors='coerce')
    tss = TimeSeriesSplit(n_splits=n_splits)

    results = {}

    # Models to train: RandomForest and LightGBM (if available)
    models = [('rfr', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]
    if LGB_AVAILABLE:
        models.append(('lgb', None))  # placeholder

    for name, estimator in models:
        maes = []
        rmses = []
        r2s = []
        for train_idx, test_idx in tss.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if name == 'lgb' and LGB_AVAILABLE:
                # build dataset for lightgbm
                dtrain = lgb.Dataset(pre.fit_transform(X_train), label=y_train)
                params = {'objective': 'regression', 'metric': 'l2', 'verbosity': -1}
                bst = lgb.train(params, dtrain, num_boost_round=100)
                preds = bst.predict(pre.transform(X_test))
            else:
                pipe = Pipeline(steps=[('preprocessor', pre), ('regressor', estimator)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            r2s.append(r2_score(y_test, preds))

        results[name] = {'mae_mean': np.mean(maes), 'rmse_mean': np.mean(rmses), 'r2_mean': np.mean(r2s)}

    # choose best by mae
    best = min(results.items(), key=lambda kv: kv[1]['mae_mean'])
    return results, best


def train_and_save(X, y, preprocessor, model='rfr', model_path=None):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"{model}.joblib")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model == 'rfr':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model == 'mlp':
        estimator = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    elif model == 'knn':
        estimator = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError('model must be "rfr" or "mlp"')

    # If model is sensitive to scaling (mlp, knn), add a scaler for numeric features inside ColumnTransformer
    if model in ('mlp', 'knn'):
        # modify preprocessor to scale numeric features
        # find numeric features from the preprocessor transformers (hack: recreate)
        # For simplicity, we'll build a new ColumnTransformer here similar to preprocess()
        numeric_features = [t for t in preprocessor.transformers_[0][2]] if hasattr(preprocessor, 'transformers_') else []
        # Build scaled numeric transformer
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        # attempt to extract columns from original preprocessor
        try:
            num_cols = preprocessor.transformers_[0][2]
            cat_cols = preprocessor.transformers_[1][2]
            new_pre = ColumnTransformer(transformers=[('num', numeric_transformer, list(num_cols)), ('cat', categorical_transformer, list(cat_cols))])
        except Exception:
            new_pre = preprocessor
        pipeline = Pipeline(steps=[('preprocessor', new_pre), ('regressor', estimator)])
    else:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', estimator)])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(pipeline, model_path)

    return {'model_path': model_path, 'mae': mae, 'r2': r2}


def recommend_sustainable(df_row: pd.Series):
    # Simple rule-based recommender:
    # - If região Norte or Nordeste and alta precipitação média -> hidrelétrica/biomassa maybe viable
    # - If temperatura média alta and muita irradiação esperada (proxy: baixa precipitação) -> solar
    # - If vento médio alto -> eólica
    # This is a heuristic placeholder; can be improved with cost/insolation/wind maps.
    recs = []
    reg = str(df_row.get('Regiao', '')).lower()
    precip = float(df_row.get('PRECIPITACAO TOTAL, MENSAL (AUT)(mm)') or 0)
    temp = float(df_row.get('TEMPERATURA MEDIA, MENSAL (AUT)(°C)') or 0)
    vento = float(df_row.get('VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)') or 0)

    if vento >= 6.0:
        recs.append('Eólica')
    if precip >= 200 and reg in ('norte', 'nordeste'):
        recs.append('Hidrelétrica / Pequenas Centrais Hidrelétricas (PCH)')
    if precip < 100 and temp >= 22:
        recs.append('Solar fotovoltaica')
    if not recs:
        recs.append('Solar fotovoltaica (padrão)')
    return '; '.join(recs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rfr', 'mlp', 'knn'], default='rfr', help='Model type to train')
    args = parser.parse_args()

    print('Carregando dados...')
    df = load_data()
    print('Linhas:', len(df))
    X, y, preprocessor = preprocess(df)
    print('Treinando modelo...', args.model)
    stats = train_and_save(X, y, preprocessor, model=args.model)
    print('Treino finalizado. MAE:', stats['mae'], 'R2:', stats['r2'], 'saved:', stats['model_path'])

    # Gerar recomendações simples para cada linha e salvar amostra
    df_sample = df.dropna(subset=['MesAno', 'Regiao']).copy()
    df_sample['Recomendacao'] = df_sample.apply(recommend_sustainable, axis=1)
    out_rec = os.path.join(MODEL_DIR, 'recomendacoes_amostra.csv')
    df_sample[['MesAno', 'Regiao', 'Consumo', 'TEMPERATURA MEDIA, MENSAL (AUT)(°C)', 'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)', 'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)', 'Recomendacao']].head(200).to_csv(out_rec, index=False, sep=';', decimal=',')
    print('Recomendações de amostra salvas em', out_rec)
