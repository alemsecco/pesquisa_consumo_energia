import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

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


def preprocess(df: pd.DataFrame):
    # Drop rows without MesAno or Regiao
    df = df.dropna(subset=['MesAno', 'Regiao']).copy()

    # Parse MesAno to extract year and month
    df['MesAno'] = df['MesAno'].astype(str)
    df['Ano'] = df['MesAno'].str.slice(0,4).astype(int)
    df['Mes'] = df['MesAno'].str.slice(5,7).astype(int)

    # Target: Consumo
    df['Consumo'] = pd.to_numeric(df['Consumo'], errors='coerce')

    # Feature columns: Ano, Mes, Regiao (one-hot), TEMPERATURA MEDIA, PRECIPITACAO, PRESSAO, VENTO
    features = []
    features += ['Ano', 'Mes']
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


def train_and_save(X, y, preprocessor, model_path=os.path.join(MODEL_DIR, 'rfr.joblib')):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, model_path)

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
    print('Carregando dados...')
    df = load_data()
    print('Linhas:', len(df))
    X, y, preprocessor = preprocess(df)
    print('Treinando modelo...')
    stats = train_and_save(X, y, preprocessor)
    print('Treino finalizado. MAE:', stats['mae'], 'R2:', stats['r2'])

    # Gerar recomendações simples para cada linha e salvar amostra
    df_sample = df.dropna(subset=['MesAno', 'Regiao']).copy()
    df_sample['Recomendacao'] = df_sample.apply(recommend_sustainable, axis=1)
    out_rec = os.path.join(MODEL_DIR, 'recomendacoes_amostra.csv')
    df_sample[['MesAno', 'Regiao', 'Consumo', 'TEMPERATURA MEDIA, MENSAL (AUT)(°C)', 'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)', 'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)', 'Recomendacao']].head(200).to_csv(out_rec, index=False, sep=';', decimal=',')
    print('Recomendações de amostra salvas em', out_rec)
