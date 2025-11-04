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

# Constants
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
MODEL_DIR = os.path.join(BASE, 'treino_modelo', 'modelo')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
    return df


def clean_numeric(s):
    """Clean numeric strings with . thousands and , decimal."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    # handle string columns with . thousands and , decimal
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
    # convert MesAno to datetime for sorting
    df['MesAno_dt'] = pd.to_datetime(df['MesAno'] + '-01', errors='coerce')
    df = df.sort_values(['Regiao', 'MesAno_dt'])
    
    # Create lag features by region
    df['lag1'] = df.groupby('Regiao')['Consumo'].shift(1)
    df['lag12'] = df.groupby('Regiao')['Consumo'].shift(12)
    df['roll3'] = df.groupby('Regiao')['Consumo'].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    return df


def recommend_sustainable(df_row: pd.Series):
    """Simple rule-based recommender:
    - If região Norte or Nordeste and alta precipitação média -> hidrelétrica/biomassa maybe viable
    - If temperatura média alta and muita irradiação esperada (proxy: baixa precipitação) -> solar
    - If vento médio alto -> eólica
    This is a heuristic placeholder; can be improved with cost/insolation/wind maps."""
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
    
    # Clean numeric columns
    df['Consumo'] = clean_numeric(df['Consumo'])
    num_cols = ['PRECIPITACAO TOTAL, MENSAL (AUT)(mm)', 
                'PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)',
                'TEMPERATURA MEDIA, MENSAL (AUT)(°C)', 
                'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    # Add lag features
    df = add_lag_features(df)
    
    # Extract year/month
    df['MesAno'] = df['MesAno'].astype(str)
    df['Ano'] = df['MesAno'].str.slice(0,4).astype(int)
    df['Mes'] = df['MesAno'].str.slice(5,7).astype(int)
    
    # Define features for model
    numeric_features = ['Ano', 'Mes', 'lag1', 'lag12', 'roll3'] + [
        c for c in num_cols if c in df.columns
    ]
    categorical_features = ['Regiao']
    
    # Build preprocessor
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
    
    # Build model pipeline
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
    
    # Train/test split by time
    df = df.sort_values('MesAno_dt')
    train_idx = df.index[:int(len(df)*0.8)]
    test_idx = df.index[int(len(df)*0.8):]
    
    # Prepare X,y
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols]
    y = df['Consumo']
    
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    
    # Train model
    print('Training model...')
    pipe.fit(X_train, y_train)
    
    # Evaluate
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f'Test MAE: {mae:,.2f}')
    print(f'Test RMSE: {rmse:,.2f}')
    print(f'Test R²: {r2:.4f}')
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'model.joblib')
    joblib.dump(pipe, model_path)
    print(f'Model saved to {model_path}')
    
    # Generate recommendations for sample
    print('\nGenerating recommendations sample...')
    df['Recomendacao'] = df.apply(recommend_sustainable, axis=1)
    
    # Save recommendations sample
    sample_cols = ['MesAno', 'Regiao', 'Consumo'] + num_cols + ['Recomendacao']
    sample_path = os.path.join(MODEL_DIR, 'recomendacoes_amostra.csv')
    df[sample_cols].head(200).to_csv(sample_path, index=False, sep=';', decimal=',')
    print(f'Recommendations sample saved to {sample_path}')
    
    # Save test predictions
    df.loc[test_idx, 'Pred_Consumo'] = preds
    predictions_path = os.path.join(MODEL_DIR, 'predictions.csv')
    df.loc[test_idx, ['MesAno', 'Regiao', 'Consumo', 'Pred_Consumo']].to_csv(
        predictions_path, index=False, sep=';', decimal=','
    )
    print(f'Test predictions saved to {predictions_path}')


if __name__ == '__main__':
    train_model()