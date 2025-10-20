# -*- coding: utf-8 -*-
"""
Pipeline completo para predição da taxa de evasão em cursos de Licenciatura em Letras (INEP).
Inclui:
- Leitura do CSV
- Separação de variáveis numéricas e categóricas
- Cálculo da taxa de evasão
- Modelos de Regressão (RandomForestRegressor, MLPRegressor)
- Modelos de Classificação (LogisticRegression, RandomForestClassifier, MLPClassifier)
- Métricas completas (MAE, RMSE, R², Accuracy, Precision, Recall, F1, ROC-AUC)
- Salvamento de arrays e modelos
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Lê CSV do INEP (padrão: separador ',')."""
    return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols, low_memory=False)

def compute_taxa_evasao(df,
                        desv_col='QT_SIT_DESVINCULADO',
                        trans_col='QT_SIT_TRANSFERIDO',
                        ing_col='QT_ING', 
                        fal_col='QT_SIT_FALECIDO', 
                        out_col='taxa_evasao',
                        min_mat=1):
    """
    Cria a variável taxa_evasao = (desvinculado + transferido) / (ingressantes - falecidos).
    Os valores são retornados em porcentagem (0-100).
    """
    df = df.copy()
    
    # Tratamento de colunas numéricas
    for c in [desv_col, trans_col, ing_col, fal_col]:
        # Converte para numérico e trata NaN como 0 para o cálculo
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Cálculo da Evasão: (Desvinculado + Transferido) / (Ingressantes - Falecidos)
    numerator = df[desv_col] + df[trans_col]
    denominator = df[ing_col] - df[fal_col]
    
    # Previne divisão por zero e filtra bases pequenas (denominador < min_mat)
    valid_denominator = (denominator > 0) & (denominator >= min_mat)
    df[out_col] = np.where(valid_denominator, (numerator / denominator) * 100, np.nan)
    
    return df

def safe_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """Wrapper seguro para train_test_split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def build_preprocessor(df, numeric_features, categorical_features):
    """Constrói um ColumnTransformer para imputação, encoding e scaling."""
    # Garante que apenas colunas que existem no df sejam usadas
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

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
# 2) Treinamento e avaliação
# ---------------------------
# [As funções de treinamento e avaliação permanecem inalteradas]
def train_regressors(X_train, y_train, preprocessor, random_state=42):
    """Treina modelos de regressão."""
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    mlp = MLPRegressor(random_state=random_state, max_iter=500)

    models = {
        'RandomForestRegressor': Pipeline(steps=[('preproc', preprocessor), ('rf', rf)]),
        'MLPRegressor': Pipeline(steps=[('preproc', preprocessor), ('mlp', mlp)]),
    }

    for name, pipe in models.items():
        print(f"Treinando modelo de regressão: {name}")
        pipe.fit(X_train, y_train)
    return models

def evaluate_regression_models(models, X_test, y_test):
    """Avalia modelos de regressão."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rows.append({'Modelo': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        print(f"[{name}] MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    return pd.DataFrame(rows)

# ---------------------------
# 3) Classificação binária
# ---------------------------

def make_binary_target_from_rate(df, rate_col='taxa_evasao', threshold=None):
    """Converte taxa contínua em rótulo binário (> mediana)."""
    if threshold is None:
        threshold = df[rate_col].median(skipna=True)
    return (df[rate_col] > threshold).astype(int), threshold

def train_classifiers(X_train, y_train, preprocessor, random_state=42):
    """Treina modelos de classificação."""
    models = {
        'LogisticRegression': Pipeline(steps=[('preproc', preprocessor),
                                              ('lr', LogisticRegression(max_iter=1000, random_state=random_state))]),
        'RandomForestClassifier': Pipeline(steps=[('preproc', preprocessor),
                                                  ('rfc', RandomForestClassifier(random_state=random_state, n_jobs=-1))]),
        'MLPClassifier': Pipeline(steps=[('preproc', preprocessor),
                                         ('mlp', MLPClassifier(max_iter=500, random_state=random_state))]),
    }

    for name, pipe in models.items():
        print(f"Treinando modelo de classificação: {name}")
        pipe.fit(X_train, y_train)
    return models

def evaluate_classification_models(models, X_test, y_test):
    """Avalia modelos de classificação."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = np.nan

        scores = {
            'Modelo': name,
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC_AUC': roc_auc
        }
        print(f"[{name}] Acc={scores['Acurácia']:.4f}, Prec={scores['Precisão']:.4f}, "
              f"Recall={scores['Recall']:.4f}, F1={scores['F1']:.4f}, ROC_AUC={scores['ROC_AUC']:.4f}")
        rows.append(scores)
    return pd.DataFrame(rows)

# ---------------------------
# 4) Execução principal
# ---------------------------

if __name__ == "__main__":
    arquivo = "MICRODADOS_CADASTRO_CURSOS_2023.CSV"
    target_col = 'taxa_evasao'

    if not os.path.exists(arquivo):
        raise SystemExit(f"Arquivo não encontrado: {arquivo}")

    # -------------------------
    # Leitura e Filtro (Licenciatura em Letras)
    # -------------------------
    df = read_inep_csv(arquivo)
    
    CURSO_COL = 'NO_CURSO'
    GRAU_COL = 'TP_GRAU_ACADEMICO' # 2 = Licenciatura
    
    # Filtro 1: Apenas cursos de Licenciatura
    df_filtered = df[df[GRAU_COL] == 2].copy()
    
    # Filtro 2: Apenas cursos que contenham 'LETRAS' no nome (case-insensitive)
    df_filtered = df_filtered[
        df_filtered[CURSO_COL].astype(str).str.contains('LETRAS', case=False, na=False)
    ].copy()
    
    # Colunas removidas após o filtro (agora são constantes)
    df_filtered.drop(columns=[CURSO_COL, GRAU_COL], inplace=True, errors='ignore')
    
    # -------------------------
    # Cálculo da variável alvo e tratamento de NA
    # -------------------------
    df_filtered = compute_taxa_evasao(df_filtered)
    df_filtered = df_filtered.dropna(subset=[target_col]).reset_index(drop=True)
    
    print("✅ CSV lido e filtrado para Licenciatura em Letras com sucesso!")
    print(f"Registros após filtro e remoção de NA: {len(df_filtered)}\n")

    # -------------------------
    # LIMPEZA E SEPARAÇÃO DE FEATURES (NOVAS REGRAS)
    # -------------------------

    # 1. Colunas a serem descartadas
    columns_to_drop = ['NU_ANO_CENSO', 'SG_UF']
    columns_to_drop.extend([col for col in df_filtered.columns if col.startswith('NO_')])
    
    # Remove as colunas de drop
    df_filtered.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 2. e 3. Classificação de features
    all_features = [c for c in df_filtered.columns if c != target_col]
    
    # 2. Numéricas: Apenas colunas que começam com 'QT_'
    numeric_features = [col for col in all_features if col.startswith('QT_')]
    
    # 3. Categóricas: Todas as outras
    categorical_features = [col for col in all_features if col not in numeric_features]

    # Converte explicitamente as categóricas para 'object' (string) para garantir que o OneHotEncoder funcione corretamente.
    # Isso evita que códigos numéricos (CO_, IN_, TP_) sejam tratados como números contínuos.
    for col in categorical_features:
        df_filtered[col] = df_filtered[col].astype(str)

    print(f"🔢 Features Numéricas (QT_): {len(numeric_features)}")
    print(f"🔠 Features Categóricas (Restantes): {len(categorical_features)}\n")
    
    # -------------------------
    # Preparação para modelagem
    # -------------------------

    X = df_filtered[numeric_features + categorical_features]
    y = df_filtered[target_col].astype(float)

    preprocessor, _, _ = build_preprocessor(
        df_filtered, numeric_features, categorical_features
    )

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    # -------------------------
    # Modelos de Regressão
    # -------------------------
    print("\n🚀 Treinando modelos de regressão...")
    models_reg = train_regressors(X_train, y_train, preprocessor)
    reg_results = evaluate_regression_models(models_reg, X_test, y_test)

    # -------------------------
    # Modelos de Classificação
    # -------------------------
    print("\n🚀 Reorientando para Classificação (taxa > mediana)...")
    y_bin, threshold = make_binary_target_from_rate(df_filtered)
    print(f"Threshold binário (mediana da taxa): {threshold:.4f}")

    # Re-split usando a variável alvo binária
    Xc_train, Xc_test, yc_train, yc_test = safe_train_test_split(
        X, y_bin, stratify=y_bin
    )

    models_clf = train_classifiers(Xc_train, yc_train, preprocessor)
    clf_results = evaluate_classification_models(models_clf, Xc_test, yc_test)

    # -------------------------
    # Resultados e salvamento
    # -------------------------
    print("\n📊 Resultados - Regressão:")
    print(reg_results)

    print("\n📊 Resultados - Classificação:")
    print(clf_results)

    os.makedirs("models_letras", exist_ok=True)
    for name, mdl in {**models_reg, **models_clf}.items():
        joblib.dump(mdl, f"models_letras/{name}.joblib")
        print(f"Modelo salvo: models_letras/{name}.joblib")

    print("\n✅ Pipeline completo executado com sucesso para Licenciatura em Letras e limpeza rigorosa de features!")
