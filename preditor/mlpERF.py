# -*- coding: utf-8 -*-
"""
Pipeline para predição da taxa de evasão em cursos de Licenciatura em Letras (INEP).
- Leitura dos microdados (CSV ; separated)
- Filtragem por curso "Letras"
- Construção da variável alvo: taxa_evasao = (QT_SIT_DESVINCULADO + QT_SIT_TRANSFERIDO) / QT_MAT
- Pré-processamento (imputação, encoding, scaling)
- Modelagem:
    * Regressão: RandomForestRegressor, MLPRegressor
    * Classificação (opcional): transformar taxa em binária (>threshold) e testar LogisticRegression, RandomForestClassifier, MLPClassifier
- Avaliação e salvamento de modelos
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
# 1) Funções utilitárias
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
# 2) Pré-processamento
# ---------------------------

def build_preprocessor(df, numeric_features=None, categorical_features=None):
    """Constrói ColumnTransformer para imputação, encoding e scaling."""
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

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

    return preprocessor, numeric_features, categorical_features

# ---------------------------
# 3) Modelagem
# ---------------------------

def train_regressors(X_train, y_train, preprocessor, random_state=42):
    """Treina RandomForestRegressor e MLPRegressor."""
    pipelines = {}

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_pipeline = Pipeline(steps=[('preproc', preprocessor), ('rf', rf)])
    rf_pipeline.fit(X_train, y_train)
    pipelines['rf'] = rf_pipeline

    mlp = MLPRegressor(random_state=random_state, max_iter=500)
    mlp_pipeline = Pipeline(steps=[('preproc', preprocessor), ('mlp', mlp)])
    mlp_pipeline.fit(X_train, y_train)
    pipelines['mlp'] = mlp_pipeline

    return pipelines

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

# ---------------------------
# 4) Classificação (opcional)
# ---------------------------

def make_binary_target_from_rate(df, rate_col='taxa_evasao', threshold=None):
    """Converte taxa contínua em binária (alto risco / baixo risco)."""
    if threshold is None:
        threshold = df[rate_col].median(skipna=True)
    return (df[rate_col] > threshold).astype(int), threshold

def train_classifiers(X_train, y_train, preprocessor, random_state=42):
    """Treina LogisticRegression, RandomForestClassifier e MLPClassifier."""
    pipelines = {}

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_pipe = Pipeline(steps=[('preproc', preprocessor), ('lr', lr)])
    lr_pipe.fit(X_train, y_train)
    pipelines['lr'] = lr_pipe

    rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rfc_pipe = Pipeline(steps=[('preproc', preprocessor), ('rfc', rfc)])
    rfc_pipe.fit(X_train, y_train)
    pipelines['rfc'] = rfc_pipe

    mlp = MLPClassifier(max_iter=500, random_state=random_state)
    mlp_pipe = Pipeline(steps=[('preproc', preprocessor), ('mlp', mlp)])
    mlp_pipe.fit(X_train, y_train)
    pipelines['mlp_clf'] = mlp_pipe

    return pipelines

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
        print(f"[{name}] acc={scores['accuracy']:.4f} prec={scores['precision']:.4f} rec={scores['recall']:.4f} f1={scores['f1']:.4f} roc_auc={scores['roc_auc']}")
        rows.append(scores)
    return pd.DataFrame(rows)

# ---------------------------
# 5) Exemplo de uso
# ---------------------------

if __name__ == "__main__":
    csv_paths = ["L_T_MICRODADOS_CADASTRO_CURSOS_2023.csv"]

    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            print("Lendo", p)
            dfs.append(read_inep_csv(p))
    if len(dfs) == 0:
        raise SystemExit("Nenhum CSV encontrado.")
    df = pd.concat(dfs, ignore_index=True)

    df = filter_letras(df, course_col='NO_CURSO')

    df = compute_taxa_evasao(df, min_mat=5)
    df = df.dropna(subset=['taxa_evasao']).reset_index(drop=True)

    numeric_features = [c for c in ['QT_SIT_DESVINCULADO','QT_SIT_TRANSFERIDO','QT_MAT'] if c in df.columns]
    categorical_features = [c for c in [
        'CO_REGIAO', 'CO_UF', 'CO_MUNICIPIO', 'IN_CAPITAL',
        'TP_REDE', 'TP_CATEGORIA_ADMINISTRATIVA',
        'TP_ORGANIZACAO_ACADEMICA', 'TP_MODALIDADE_ENSINO'
    ] if c in df.columns]

    preprocessor, numeric_features, categorical_features = build_preprocessor(
        df, numeric_features=numeric_features, categorical_features=categorical_features)

    X = df[numeric_features + categorical_features].copy()
    y_reg = df['taxa_evasao'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    print("Treinando modelos de regressão...")
    models_reg = train_regressors(X_train, y_train, preprocessor)

    print("Avaliando regressão...")
    reg_res = evaluate_regression_models(models_reg, X_test, y_test)
    print(reg_res)

    y_bin, thresh = make_binary_target_from_rate(df, rate_col='taxa_evasao')
    print("Threshold binário:", thresh)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

    print("Treinando classificadores...")
    models_clf = train_classifiers(Xc_train, yc_train, preprocessor)

    print("Avaliando classificadores...")
    clf_res = evaluate_classification_models(models_clf, Xc_test, yc_test)
    print(clf_res)

    # salvar modelos
    os.makedirs('models', exist_ok=True)
    for name, mdl in {**models_reg, **models_clf}.items():
        fname = os.path.join('models', f'{name}.joblib')
        joblib.dump(mdl, fname)
        print("Salvo:", fname)

    print("Pipeline concluído.")
