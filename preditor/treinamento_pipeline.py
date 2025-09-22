# -*- coding: utf-8 -*-
"""
Pipeline para predição da taxa de evasão em cursos de Licenciatura em Letras (INEP).

Este script foi refatorado para:
- Permitir a seleção de modelos (ex: 'rf', 'mlp') como parâmetro.
- Salvar um único arquivo de 'artefatos' (modelos, pré-processador, features)
  para ser usado por um script de previsão separado.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------
# 1) Funções utilitárias (sem alterações)
# ---------------------------

def read_inep_csv(path, sep=';', encoding='latin1', usecols=None):
    """Lê CSV do INEP com separador ponto e vírgula por padrão."""
    return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)

def filter_letras(df, course_col='NO_CURSO'):
    """Filtra linhas correspondentes a cursos de Letras (heurística)."""
    mask = df[course_col].astype(str).str.upper().str.contains('LETRAS')
    return df[mask].copy()

def compute_taxa_evasao(df,
                        desv_col='QT_SIT_DESVINCULADO',
                        trans_col='QT_SIT_TRANSFERIDO',
                        mat_col='QT_MAT',
                        out_col='taxa_evasao',
                        min_mat=1):
    """Cria coluna taxa_evasao = (desvinculado + transferido) / matriculados."""
    df = df.copy()
    for c in [desv_col, trans_col, mat_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df[out_col] = (df[desv_col] + df[trans_col]) / df[mat_col].replace({0: np.nan})
    df.loc[df[mat_col] < min_mat, out_col] = np.nan
    return df

# ---------------------------
# 2) Pré-processamento (sem alterações)
# ---------------------------

def build_preprocessor(numeric_features, categorical_features):
    """Constrói ColumnTransformer para imputação, encoding e scaling."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    return preprocessor

# ---------------------------
# 3) Modelagem (Refatorado)
# ---------------------------

# Dicionário de configuração central para todos os modelos
MODELS_CONFIG = {
    'rf_reg': (RandomForestRegressor, {'random_state': 42, 'n_jobs': -1}),
    'mlp_reg': (MLPRegressor, {'random_state': 42, 'max_iter': 500}),
    'lr_clf': (LogisticRegression, {'max_iter': 1000, 'random_state': 42}),
    'rf_clf': (RandomForestClassifier, {'random_state': 42, 'n_jobs': -1}),
    'mlp_clf': (MLPClassifier, {'random_state': 42, 'max_iter': 500})
}

def train_models(X_train, y_train, preprocessor, model_keys):
    """
    Treina uma lista de modelos especificados por suas chaves em MODELS_CONFIG.
    """
    pipelines = {}
    for key in model_keys:
        if key not in MODELS_CONFIG:
            print(f"Aviso: Chave de modelo '{key}' não encontrada em MODELS_CONFIG. Pulando.")
            continue
        
        model_class, model_params = MODELS_CONFIG[key]
        model = model_class(**model_params)
        
        pipeline = Pipeline(steps=[('preproc', preprocessor), (key, model)])
        print(f"Treinando modelo: {key}...")
        pipeline.fit(X_train, y_train)
        pipelines[key] = pipeline
        
    return pipelines

# ---------------------------
# 4) Avaliação (sem alterações)
# ---------------------------

def evaluate_regression_models(models_dict, X_test, y_test):
    """Avalia modelos de regressão (MAE, RMSE, R2)."""
    rows = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rows.append({'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        print(f"[{name}] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
    return pd.DataFrame(rows)

def make_binary_target_from_rate(df, rate_col='taxa_evasao', threshold=None):
    """Converte taxa contínua em binária (alto risco / baixo risco)."""
    if threshold is None:
        threshold = df[rate_col].median(skipna=True)
    return (df[rate_col] > threshold).astype(int), threshold
    
def evaluate_classification_models(models_dict, X_test, y_test):
    """Avalia classificadores (accuracy, precision, recall, f1, roc_auc)."""
    rows = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        scores = {
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            scores['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            scores['roc_auc'] = np.nan
        print(f"[{name}] acc={scores['accuracy']:.4f} prec={scores['precision']:.4f} rec={scores['recall']:.4f} f1={scores['f1']:.4f} roc_auc={scores.get('roc_auc', 'N/A'):.4f}")
        rows.append(scores)
    return pd.DataFrame(rows)

# ---------------------------
# 5) Execução Principal
# ---------------------------

if __name__ == "__main__":
    # --- Carregamento e Preparação dos Dados ---
    csv_paths = ["L_T_MICRODADOS_CADASTRO_CURSOS_2023.csv"] # Use o nome do seu arquivo

    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            print("Lendo", p)
            dfs.append(read_inep_csv(p))
    if not dfs:
        raise SystemExit("Nenhum CSV encontrado. Verifique o caminho do arquivo.")
    
    df = pd.concat(dfs, ignore_index=True)
    df = filter_letras(df, course_col='NO_CURSO')
    df = compute_taxa_evasao(df, min_mat=5)
    df = df.dropna(subset=['taxa_evasao']).reset_index(drop=True)

    # --- Definição das Features ---
    # Adicione ou remova colunas conforme necessário
    numeric_features = [c for c in ['QT_MAT', 'QT_VAGAS_NOVAS_INTEGRAL'] if c in df.columns]
    categorical_features = [c for c in [
        'CO_REGIAO', 'CO_UF', 'TP_REDE', 'TP_CATEGORIA_ADMINISTRATIVA',
        'TP_ORGANIZACAO_ACADEMICA', 'TP_MODALIDADE_ENSINO'
    ] if c in df.columns]
    
    features = numeric_features + categorical_features
    X = df[features].copy()
    
    # --- Treinamento de Regressão ---
    y_reg = df['taxa_evasao'].astype(float)
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print("\n--- Treinando modelos de regressão ---")
    reg_models_to_train = ['rf_reg', 'mlp_reg'] # Defina aqui os regressores que quer treinar
    trained_reg_models = train_models(X_train, y_train_reg, preprocessor, reg_models_to_train)
    
    print("\n--- Avaliando regressão ---")
    reg_res = evaluate_regression_models(trained_reg_models, X_test, y_test_reg)
    print(reg_res)

    # --- Treinamento de Classificação ---
    y_bin, threshold = make_binary_target_from_rate(df, rate_col='taxa_evasao')
    print(f"\nThreshold para classificação (mediana da taxa de evasão): {threshold:.4f}")
    
    Xc_train, Xc_test, y_train_clf, y_test_clf = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    
    print("\n--- Treinando classificadores ---")
    clf_models_to_train = ['rf_clf', 'mlp_clf'] # Defina aqui os classificadores que quer treinar
    trained_clf_models = train_models(Xc_train, y_train_clf, preprocessor, clf_models_to_train)

    print("\n--- Avaliando classificadores ---")
    clf_res = evaluate_classification_models(trained_clf_models, Xc_test, y_test_clf)
    print(clf_res)

    # --- Salvando os artefatos para previsão ---
    artifacts = {
        'reg_models': trained_reg_models,
        'clf_models': trained_clf_models,
        'features': features,
        'classification_threshold': threshold,
        'preprocessor': preprocessor
    }
    
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    artifacts_path = os.path.join(output_dir, 'prediction_artifacts.joblib')
    joblib.dump(artifacts, artifacts_path)
    print(f"\nArtefatos de previsão salvos em: {artifacts_path}")

    print("\nPipeline de treinamento concluído.")