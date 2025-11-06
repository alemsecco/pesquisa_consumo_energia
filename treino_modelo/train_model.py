import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Constantes de caminho
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
MODEL_DIR = os.path.join(BASE, 'treino_modelo', 'modelo')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
    return df


def clean_numeric(s):
    """ limpa strings numéricas com . milhares e , decimal."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    # lida com colunas de string com . milhares e , decimal
    return pd.to_numeric(
        s.astype(str)
        .str.replace('\xa0', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False),
        errors='coerce'
    )


def add_lag_features(df: pd.DataFrame):
    df = df.copy()
    # converte MesAno para datetime para ordenação
    df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')
    df = df.sort_values(['Regiao', 'MesAno_dt'])
    
    # cria laf features por região
    df['lag1'] = df.groupby('Regiao')['Consumo'].shift(1)
    df['lag12'] = df.groupby('Regiao')['Consumo'].shift(12)
    df['roll3'] = df.groupby('Regiao')['Consumo'].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    return df


def recommend_sustainable(df_row: pd.Series):
    """ recomendações sustentáveis simples baseado em regras:
    - Se região norte ou norteste E alta precipitação média -> hidrelétrica/biomassa pode ser viável
    - Se temperatura média alta E muita irradiação esperada (proxy: baixa precipitação) -> solar
    - Se vento médio alto -> eólica
    É uma heurística simples; pode ser melhorada com mapas de custo/insolação/vento."""
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


def train_model():
    print('Loading data...')
    df = load_data()
    print('Rows:', len(df))
    
    # limpa colunas numéricas
    df['Consumo'] = clean_numeric(df['Consumo'])
    num_cols = ['PRECIPITACAO TOTAL, MENSAL (AUT)(mm)', 
                'PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)',
                'TEMPERATURA MEDIA, MENSAL (AUT)(°C)', 
                'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    # add lag features
    df = add_lag_features(df)
    
    # extrai ano/mes
    df['MesAno'] = df['MesAno'].astype(str)
    df['Ano'] = df['MesAno'].str.slice(0,4).astype(int)
    df['Mes'] = df['MesAno'].str.slice(5,7).astype(int)
    
    # define features para modelo
    numeric_features = ['Ano', 'Mes', 'lag1', 'lag12', 'roll3'] + [
        c for c in num_cols if c in df.columns
    ]
    categorical_features = ['Regiao']
    
    # constrói preprocessador
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # constrói pipeline do modelo
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=40,
        min_samples_leaf=2,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42
    )
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Treino/teste dividido por tempo
    df = df.sort_values('MesAno_dt')
    train_idx = df.index[:int(len(df)*0.8)]
    test_idx = df.index[int(len(df)*0.8):]
    
    # Prepara X,y
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols]
    y = df['Consumo']
    
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    
    # Treina modelo
    print('Training model...')
    pipe.fit(X_train, y_train)
    
    # avalia
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f'Test MAE: {mae:,.2f}')
    print(f'Test RMSE: {rmse:,.2f}')
    print(f'Test R²: {r2:.4f}')
    
    # salva modelo
    model_path = os.path.join(MODEL_DIR, 'model.joblib')
    joblib.dump(pipe, model_path)
    print(f'Model saved to {model_path}')
    
    # gera recomendações para amostra
    print('\nGenerating recommendations sample...')
    df['Recomendacao'] = df.apply(recommend_sustainable, axis=1)
    
    # salva recomendações
    sample_cols = ['MesAno', 'Regiao', 'Consumo'] + num_cols + ['Recomendacao']
    sample_path = os.path.join(MODEL_DIR, 'recomendacoes_amostra.csv')
    df[sample_cols].head(200).to_csv(sample_path, index=False, sep=';', decimal=',')
    print(f'Recommendations sample saved to {sample_path}')
    
    # salva previsões do teste
    df.loc[test_idx, 'Pred_Consumo'] = preds
    predictions_path = os.path.join(MODEL_DIR, 'predictions.csv')
    df.loc[test_idx, ['MesAno', 'Regiao', 'Consumo', 'Pred_Consumo']].to_csv(
        predictions_path, index=False, sep=';', decimal=','
    )
    print(f'Test predictions saved to {predictions_path}')


if __name__ == '__main__':
    train_model()