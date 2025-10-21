# 
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# 
# Import bibliotecas principais para manipulação de dados e ML
import pandas as pd
import numpy as np
import warnings
import os

# Modelos de classificação e regressão 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

# Ferramentas de pré-processamento e criação de pipelines
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit

# Ferramentas para avaliação de modelos (hold-out e k-fold)
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
    make_scorer
)

# Ignoro avisos que não interferem nos resultados (pra deixar a saída limpa)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

print("Bibliotecas importadas com sucesso.")

# 
# 2. FUNÇÕES AUXILIARES
# 

def load_csv_to_df(path, encoding=None, sep=','):
    """Carrega um arquivo CSV para um pandas.DataFrame.
    Tenta algumas codificações comuns se `encoding` não for fornecido.
    Retorna DataFrame ou None em caso de erro.
    """
    if not os.path.exists(path):
        print(f"Erro: arquivo '{path}' não encontrado.")
    # (script)

    encodings_to_try = [encoding] if encoding else ['utf-8', 'latin1', 'cp1252']
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep)
            print(f"CSV carregado com sucesso: '{path}' (encoding={enc})")
            return df
        except Exception as e:
            last_err = e
            # tenta próxima codificação

    print(f"Falha ao ler '{path}' com as codificações testadas {encodings_to_try}. Erro: {last_err}")
    return None

def specificity_score(y_true, y_pred):
    """Calcula especificidade = TN / (TN + FP)."""
    # sklearn não possui essa métrica, então implementei manualmente
    if len(np.unique(y_true)) > 2:
        # Versão multiclasse — calcula média por classe
        cm = confusion_matrix(y_true, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specificity = np.nan_to_num(tn / (tn + fp))
        return np.mean(specificity)
    else:
        # Versão binária simples
        cm_flat = confusion_matrix(y_true, y_pred).ravel()
        if len(cm_flat) == 4:
            tn, fp, fn, tp = cm_flat
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            return 0.0

def get_classification_metrics(y_true, y_pred):
    """Calcula as métricas da parte de classificação."""
    # Retorna todas as colunas das Tabelas A e B
    return {
        'Taxa de Acerto (%)': accuracy_score(y_true, y_pred) * 100,
        'F1 (%)': f1_score(y_true, y_pred, average='macro') * 100,
        'Precisão (%)': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'Sensibilidade (%)': recall_score(y_true, y_pred, average='macro') * 100,
        'Especificidade (%)': specificity_score(y_true, y_pred) * 100
    }

def get_regression_metrics(y_true, y_pred):
    """Calcula as métricas da parte de regressão."""
    # Métricas das Tabelas C e D
    return {
        'Coeficiente de Determinação (R2)': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

print("Funções auxiliares definidas.")

# 
# 3. PREPARAÇÃO DOS DADOS
# 

# Dataset único: regressão para 'Consumo'
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'dados', 'consumo_vs_temperatura_por_regiao.csv')
df = load_csv_to_df(data_path, sep=';')

df_reg = df

# --- Pré-processamento da Regressão ---
if df_reg is not None:
    print(f"Dataset de Regressão carregado: {df_reg.shape[0]} linhas, {df_reg.shape[1]} colunas")
    # target = 'Consumo'
    target_reg = 'Consumo'
    if target_reg not in df_reg.columns:
        print(f"ERRO: coluna alvo '{target_reg}' não encontrada em {data_path}. Verifique o arquivo.")
        df_reg = None
    else:
        # feature engineering básica: extrair Ano and Mes de MesAno se existe
        df_reg = df_reg.copy()
        if 'MesAno' in df_reg.columns:
            try:
                df_reg['Ano'] = df_reg['MesAno'].astype(str).str.slice(0,4).astype(int)
                df_reg['Mes'] = df_reg['MesAno'].astype(str).str.slice(5,7).astype(int)
            except Exception:
                pass

            # limpar target 'Consumo': remove separador de milhar e converte virgula decimal
            df_reg[target_reg] = df_reg[target_reg].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            y_reg = pd.to_numeric(df_reg[target_reg], errors='coerce')
            # Drop linhas onde target é NaN
            n_before = len(df_reg)
            mask = y_reg.notna()
            df_reg = df_reg.loc[mask].reset_index(drop=True)
            y_reg = y_reg.loc[mask].reset_index(drop=True)
            n_after = len(df_reg)
            print(f"Linhas com target válido: {n_after} (removidas {n_before - n_after} linhas com target inválido)")
        # escolhe colunas de features: medições numéricas + Ano, Mes, Regiao
        candidate_cols = list(df_reg.columns)
        # drop target e MesAno
        for c in [target_reg, 'MesAno']:
            if c in candidate_cols:
                candidate_cols.remove(c)

        # mantém um conjunto sensato: Ano, Mes, Regiao, e colunas numéricas de medição
        keep = [c for c in candidate_cols if c in ('Ano', 'Mes', 'Regiao') or df_reg[c].dtype in (float, int) or df_reg[c].dtype == 'object']
        X_reg = df_reg[keep].copy()

        # pré-processador: numérico imputer+scaler + onehot para Regiao
        numeric_cols = [c for c in X_reg.columns if c not in ('Regiao',) and pd.api.types.is_numeric_dtype(X_reg[c])]
        categorical_cols = [c for c in X_reg.columns if c == 'Regiao' or not pd.api.types.is_numeric_dtype(X_reg[c])]

        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor_reg = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)])
        print('Pré-processamento configurado. Numeric cols:', numeric_cols, 'Categorical cols:', categorical_cols)

# 
# 4. DEFINIÇÃO DOS MODELOS
# 

# Aqui defino todos os algoritmos que serão testados
# A ideia é facilitar a execução automática de todos eles

# --- Modelos de Classificação ---
estimators_class = [('dt', DecisionTreeClassifier(random_state=42)), ('knn', KNeighborsClassifier())]
voting_estimators_class = [('rf', RandomForestClassifier(random_state=42)), ('svm', SVC(probability=True, random_state=42)), ('nb', GaussianNB())]

models_class = {
    "KNN": KNeighborsClassifier(), 
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(), 
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    "SVM": SVC(random_state=42),
    "Ensemble (somatória)": VotingClassifier(estimators=voting_estimators_class, voting='soft'),
    "Random Forest": RandomForestClassifier(random_state=42), 
    "Bagging": BaggingClassifier(random_state=42),
    "Boosting": AdaBoostClassifier(random_state=42),
    "Stacking": StackingClassifier(estimators=estimators_class, final_estimator=RandomForestClassifier(random_state=42)),
    "Blending": StackingClassifier(estimators=estimators_class, final_estimator=RandomForestClassifier(random_state=42), cv=2),
    "Adicional": GradientBoostingClassifier(random_state=42)
}

# --- Modelos de Regressão ---
estimators_reg = [('dt', DecisionTreeRegressor(random_state=42)), ('knn', KNeighborsRegressor())]
voting_estimators_reg = [('rf', RandomForestRegressor(random_state=42)), ('svr', SVR()), ('lr', LinearRegression())]

models_reg = {
    "Regressão Linear": LinearRegression(), 
    "KNN": KNeighborsRegressor(),
    "Árvore de Decisão": DecisionTreeRegressor(random_state=42), 
    "MLP": MLPRegressor(max_iter=1000, random_state=42),
    "SVM": SVR(), 
    "Ensemble (Média)": VotingRegressor(estimators=voting_estimators_reg),
    "Random Forest": RandomForestRegressor(random_state=42), 
    "Bagging": BaggingRegressor(random_state=42),
    "Boosting": AdaBoostRegressor(random_state=42),
    "Stacking": StackingRegressor(estimators=estimators_reg, final_estimator=RandomForestRegressor(random_state=42)),
    "Blending": StackingRegressor(estimators=estimators_reg, final_estimator=RandomForestRegressor(random_state=42), cv=2),
    "Adicional": GradientBoostingRegressor(random_state=42)
}
print("Modelos definidos.")

# 
# 5. EXECUÇÃO DOS EXPERIMENTOS
# 

# Aqui são geradas as 4 tabelas 
# Cada bloco executa todos os modelos e armazena as métricas

if df_class is not None:
    # --- Tabela A: Classificação com Hold-out ---
    print("\n--- Processando Tabela A: Classificação com Hold-out (65/35) ---")
    results_A = []
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.35, random_state=42, stratify=y_class)
    for name, model in models_class.items():
        # Uso de pipeline pra aplicar normalização automaticamente
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_class), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        # Calculo as métricas 
        metrics = get_classification_metrics(y_test, y_pred)
        metrics['Indutor'] = name
        results_A.append(metrics)
        print(f"  - {name}: Concluído")  # comentário rápido: indica progresso
    df_results_A = pd.DataFrame(results_A)[['Indutor', 'Taxa de Acerto (%)', 'F1 (%)', 'Precisão (%)', 'Sensibilidade (%)', 'Especificidade (%)']]

    # --- Tabela B: Classificação com Validação Cruzada (5 folds) ---
    print("\n--- Processando Tabela B: Classificação com Validação Cruzada (5 folds) ---")
    results_B = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring_class = {
        'accuracy': 'accuracy', 
        'f1_macro': 'f1_macro', 
        'precision_macro': 'precision_macro', 
        'recall_macro': 'recall_macro', 
        'specificity': make_scorer(specificity_score)
    }
    for name, model in models_class.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor_class), ('classifier', model)])
        scores = cross_validate(pipeline, X_class, y_class, cv=kfold, scoring=scoring_class)
        results_B.append({
            'Indutor': name,
            'Taxa de Acerto (%)': scores['test_accuracy'].mean() * 100,
            'F1 (%)': scores['test_f1_macro'].mean() * 100,
            'Precisão (%)': scores['test_precision_macro'].mean() * 100,
            'Sensibilidade (%)': scores['test_recall_macro'].mean() * 100,
            'Especificidade (%)': scores['test_specificity'].mean() * 100
        })
        print(f"  - {name}: Concluído")
    df_results_B = pd.DataFrame(results_B)[['Indutor', 'Taxa de Acerto (%)', 'F1 (%)', 'Precisão (%)', 'Sensibilidade (%)', 'Especificidade (%)']]
        # (fim do script)
if df_reg is not None:
    # uso de TimeSeriesSplit pra cross-validation temporal
    print("\n--- Avaliando modelos de regressão com TimeSeriesSplit (5 folds) ---")
    results = []
    tss = TimeSeriesSplit(n_splits=5)
    for name, model in models_reg.items():
        print(f"Treinando/avaliando: {name}")
        pipe = Pipeline(steps=[('preprocessor', preprocessor_reg), ('regressor', model)])
        try:
            scores = cross_validate(pipe, X_reg, y_reg, cv=tss, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], error_score='raise')
            mae = -scores['test_neg_mean_absolute_error'].mean()
            rmse = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
            r2m = scores['test_r2'].mean()
        except Exception as e:
            print(f"  Erro ao avaliar {name}: {e}")
            mae = np.nan
            rmse = np.nan
            r2m = np.nan

        results.append({'Indutor': name, 'R2': r2m, 'RMSE': rmse, 'MAE': mae})

    df_results = pd.DataFrame(results).sort_values('MAE')
    out_csv = os.path.join(base_dir, 'treino_modelo', 'resultados_modelos_regressao.csv')
    df_results.to_csv(out_csv, index=False, sep=';', decimal=',')
    print('\nResultados salvos em', out_csv)

# 
# 6. EXIBIÇÃO DOS RESULTADOS
# 

# Exibo todas as tabelas no formato exigido pelo enunciado
# Comentário rápido: essas saídas correspondem às quatro tabelas

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print("\n\n" + "="*80)
print("RESULTADOS FINAIS")
print("="*80)

if 'df_results_A' in locals():
    print("\n\nTABELA A: Classificação com protocolo Hold-out (65% para treinamento e 35% para teste)")
    print(df_results_A.to_string(index=False))
    print("\n\nTABELA B: Classificação com protocolo experimental validação cruzada com 5 folds")
    print(df_results_B.to_string(index=False))

if 'df_results_C' in locals():
    print("\n\nTABELA C: Regressão com protocolo Hold-out (65% para treinamento e 35% para teste)")
    print(df_results_C.to_string(index=False))
    print("\n\nTABELA D: Regressão com protocolo experimental validação cruzada com 5 folds")
    print(df_results_D.to_string(index=False))