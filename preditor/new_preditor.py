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
# 1) Funções utilitárias
# ---------------------------

def compute_taxa_evasao(df,
                        desv_col='QT_SIT_DESVINCULADO',
                        trans_col='QT_SIT_TRANSFERIDO',
                        mat_col='QT_MAT',
                        out_col='taxa_evasao',
                        min_mat=1):
    """Cria a variável taxa_evasao = (desvinculado + transferido) / matriculados."""
    df = df.copy()
    # Converte para numérico de forma segura e preenche NaNs com 0 para o cálculo
    for c in [desv_col, trans_col, mat_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
    # Cálculo da taxa, substituindo 0 matriculados por NaN para evitar divisão por zero
    df[out_col] = (df[desv_col] + df[trans_col]) / df[mat_col].replace({0: np.nan})
    
    # Define taxa como NaN se o número de matriculados for menor que o mínimo
    df.loc[df[mat_col] < min_mat, out_col] = np.nan
    return df

def safe_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """Wrapper seguro para train_test_split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def build_preprocessor(df, numeric_features=None, categorical_features=None):
    """Constrói um ColumnTransformer para imputação, encoding e scaling."""
    # Se as listas não forem fornecidas, tenta inferir os tipos.
    if numeric_features is None:
        # Tenta incluir Int64Dtype, que é tratada como 'number'
        numeric_features = df.select_dtypes(include=[np.number, pd.Int64Dtype]).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        # A estratégia 'most_frequent' é segura para 'category'
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        # OneHotEncoder para variáveis categóricas
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop') # Garante que apenas as colunas especificadas sejam usadas

    return preprocessor, numeric_features, categorical_features

# ---------------------------
# 2) Treinamento e avaliação (Regressão)
# ---------------------------

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
# 3) Treinamento e avaliação (Classificação binária)
# ---------------------------

def make_binary_target_from_rate(df, rate_col='taxa_evasao', threshold=None):
    """Converte taxa contínua em rótulo binário (> mediana)."""
    if threshold is None:
        threshold = df[rate_col].median(skipna=True)
    # A variável binária é True (1) se a taxa de evasão for MAIOR que a mediana
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
            # Tenta calcular ROC-AUC usando probabilidades
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = np.nan # ROC-AUC não disponível para modelos que não suportam predict_proba

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
# FUNÇÃO DE CARREGAMENTO E PRÉ-PROCESSAMENTO (NOVA)
# ---------------------------

def load_and_preprocess_data(csv_file):
    """
    Carrega, filtra, aplica tipagem e prepara o DataFrame para o pipeline de ML.
    Define colunas CO_, TP_, IN_ como 'category' e QT_ como pd.Int64Dtype().
    """
    sep_char = ';' # Delimitador dos microdados do INEP

    # Listas de prefixos para definir os tipos
    prefixos_categoricos = ('CO_', 'TP_', 'IN_')
    prefixos_quantitativos = ('QT_')
    
    # Colunas de texto (nomes) que DEVEM ser removidas após a filtragem
    colunas_para_remover = [
        'NU_ANO_CENSO', 'NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO',
        'NO_CINE_ROTULO', 'NO_CINE_AREA_GERAL',
        'NO_CINE_AREA_ESPECIFICA', 'NO_CINE_AREA_DETALHADA'
    ]
    # Coluna usada para a filtragem de Letras, também removida
    colunas_para_remover_e_filtrar = colunas_para_remover + ['NO_CURSO']

    # --- 1. Determinação da tipagem (dtype_mapping) ---
    dtype_mapping = {}
    
    # Lê APENAS o cabeçalho para obter a lista completa de colunas
    try:
        temp_df = pd.read_csv(csv_file, sep=sep_char, encoding='latin1', nrows=0)
        all_cols = temp_df.columns.tolist()
    except Exception as e:
        print(f"Erro ao ler cabeçalho: {e}")
        return pd.DataFrame()

    colunas_manter = [c for c in all_cols if c not in colunas_para_remover]

    for col in all_cols:
        if col.startswith(prefixos_categoricos):
            dtype_mapping[col] = 'category'
        elif col.startswith(prefixos_quantitativos):
            # pd.Int64Dtype permite inteiros e NaN (ideal para QT_)
            dtype_mapping[col] = pd.Int64Dtype()
        else:
            # Mantém as colunas de texto (nomes) como string para leitura e filtragem
            dtype_mapping[col] = 'string' 

    # --- 2. Leitura com Tipagem e Filtragem Inicial ---
    try:
        df = pd.read_csv(
            csv_file,
            sep=sep_char,
            encoding='latin1',
            dtype=dtype_mapping,
            low_memory=False
        )
        print(f"Dados carregados com sucesso. Total de {len(df)} registros.")
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return pd.DataFrame()

    # --- 3. Filtragem de Linhas ---
    # Filtrar para Licenciatura (TP_GRAU_ACADEMICO == 2)
    df = df.loc[df['TP_GRAU_ACADEMICO'] == 2]

    # Filtrar para 'Letras' no nome do curso (case insensitive)
    df = df.loc[df['NO_CURSO'].str.contains('Letras', case=False, na=False)].copy()
    
    # --- 4. Remoção de Colunas de Texto ---
    # Remove as colunas de texto (nomes) que não serão usadas no modelo
    df = df.drop(columns=colunas_para_remover_e_filtrar, errors='ignore')
    
    return df

# ---------------------------
# 4) Execução principal
# ---------------------------

if __name__ == "__main__":
    # Substituímos o arquivo genérico pelo seu arquivo INEP
    arquivo = "MICRODADOS_CADASTRO_CURSOS_2024.CSV" 
    
    # -------------------------
    # Leitura e Filtragem (NOVA LÓGICA)
    # -------------------------
    if not os.path.exists(arquivo):
        raise SystemExit(f"Arquivo não encontrado: {arquivo}")

    df = load_and_preprocess_data(arquivo)
    
    if df.empty:
        raise SystemExit("O DataFrame está vazio após a leitura e filtragem inicial. Encerrando pipeline.")

    # -------------------------
    # Cálculo da variável alvo (Regressão)
    # -------------------------
    df = compute_taxa_evasao(df)
    # Remove linhas onde a taxa de evasão é NaN (divisão por zero ou poucos matriculados)
    df_reg = df.dropna(subset=['taxa_evasao']).reset_index(drop=True) 
    print(f"Registros válidos para Regressão após cálculo da taxa: {len(df_reg)}")

    # -------------------------
    # Preparação para modelagem
    # -------------------------
    target_col = 'taxa_evasao'
    
    # Separa as features usando os tipos de dados do DataFrame (mais seguro)
    # Apenas colunas categóricas e Int64 (numéricas) são consideradas features
    numeric_features = df_reg.select_dtypes(include=[pd.Int64Dtype, np.number]).columns.tolist()
    categorical_features = df_reg.select_dtypes(include=['category']).columns.tolist()
    
    # Remove a coluna target das features
    if target_col in numeric_features:
        numeric_features.remove(target_col)

    # Verifica se restaram features
    if not numeric_features and not categorical_features:
        raise SystemExit("Nenhuma feature numérica ou categórica restante após a limpeza. Encerrando.")


    # Define o pré-processador para todo o pipeline
    preprocessor, numeric_features, categorical_features = build_preprocessor(
        df_reg, numeric_features, categorical_features
    )

    X = df_reg[numeric_features + categorical_features]
    y = df_reg[target_col].astype(float) # Garante que o target é float

    # Salvamento de arrays e features
    os.makedirs("arrays_separados", exist_ok=True)
    pd.Series(numeric_features).to_csv("arrays_separados/numeric_features.csv", index=False)
    pd.Series(categorical_features).to_csv("arrays_separados/categorical_features.csv", index=False)
    # Salvamento dos dados brutos (antes do pré-processamento)
    np.save("arrays_separados/X_array_raw.npy", X.to_numpy())
    np.save("arrays_separados/y_array.npy", y.to_numpy())
    print("💾 Arrays brutos salvos em 'arrays_separados'\n")
    
    # Separação Treino/Teste
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
    print("\n🚀 Treinando modelos de classificação (taxa > mediana)...")
    # Usa o DataFrame original (df) para calcular a mediana global
    y_bin, threshold = make_binary_target_from_rate(df) 
    
    # Filtra o X para corresponder ao y_bin (que tem o mesmo índice do df original)
    X_clf = df[numeric_features + categorical_features] 
    
    # Remove NaNs em X_clf (se houver algum) e alinha com y_bin
    X_clf = X_clf.dropna() 
    y_bin = y_bin.loc[X_clf.index] # Alinha o target com as features limpas
    
    print(f"Threshold binário (mediana da taxa): {threshold:.4f}")
    print(f"Registros válidos para Classificação: {len(X_clf)}")

    # Separação Treino/Teste Classificação
    Xc_train, Xc_test, yc_train, yc_test = safe_train_test_split(
        X_clf, y_bin, stratify=y_bin
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

    os.makedirs("models", exist_ok=True)
    for name, mdl in {**models_reg, **models_clf}.items():
        joblib.dump(mdl, f"models/{name}.joblib")
        print(f"Modelo salvo: models/{name}.joblib")

    print("\n✅ Pipeline completo executado com sucesso!")
