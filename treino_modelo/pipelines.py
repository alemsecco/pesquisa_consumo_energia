import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

def build_tree_pipeline(preprocessor):
    """Retorna uma pipeline adequada pra modelos baseados em árvores (Random Forest). mantém imput numérico, sem escalonador."""
    # reusa colunas numéricas e categóricas do pré-processador, se disponível
    try:
        num_cols = preprocessor.transformers[0][2]
        cat_cols = preprocessor.transformers[1][2]
    except Exception:
        # fallback listas vazias
        num_cols, cat_cols = [], []

    # transformer numérico: imputação pela mediana (usa existente)
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    new_pre = ColumnTransformer(transformers=[('num', numeric_transformer, list(num_cols)), ('cat', categorical_transformer, list(cat_cols))])

    pipe = Pipeline(steps=[('preprocessor', new_pre), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    return pipe


def build_sensitive_pipeline(preprocessor, scaler='standard', use_power=False):
    """ Retorna uma pipeline pra modelos sensíveis a escalonamento (MLP, KNN). escalonador: standard ou robusto.
    use_power: se True aplica PowerTransformer (Yeo-Johnson) às features numéricas
    """
    try:
        num_cols = preprocessor.transformers[0][2]
        cat_cols = preprocessor.transformers[1][2]
    except Exception:
        num_cols, cat_cols = [], []

    if scaler == 'robust':
        scaler_obj = RobustScaler()
    else:
        scaler_obj = StandardScaler()

    steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_power:
        steps.append(('power', PowerTransformer(method='yeo-johnson')))
    steps.append(('scaler', scaler_obj))

    numeric_transformer = Pipeline(steps=steps)
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    new_pre = ColumnTransformer(transformers=[('num', numeric_transformer, list(num_cols)), ('cat', categorical_transformer, list(cat_cols))])

    pipe = Pipeline(steps=[('preprocessor', new_pre), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    return pipe
