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
- Avaliação e importância de variáveis
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1) Funções utilitárias
# ---------------------------

def read_inep_csv(path, sep=';', encoding='latin1', usecols=None):
    """Lê CSV do INEP com separador ponto e vírgula por padrão."""
    return pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)

def filter_letras(df, course_col='NO_CURSO', code_col='CO_CURSO'):
    """
    Filtra linhas correspondentes a cursos de Letras.
    Heurística: procura 'LETRAS' no nome do curso (case-insensitive) ou em outros campos.
    Ajuste se necessário.
    """
    mask = df[course_col].astype(str).str.upper().str.contains('LETRAS')
    return df[mask].copy()

def compute_taxa_evasao(df,
                       desv_col='QT_SIT_DESVINCULADO',
                       trans_col='QT_SIT_TRANSFERIDO',
                       mat_col='QT_MAT',
                       out_col='taxa_evasao',
                       min_mat=1):
    """
    Cria coluna taxa_evasao = (desvinculado + transferido) / matriculados.
    Remove (ou marca) casos com QT_MAT < min_mat para evitar divisão por zero / ruído.
    """
    df = df.copy()
    # garantir colunas numéricas
    for c in [desv_col, trans_col, mat_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df[out_col] = (df[desv_col] + df[trans_col]) / df[mat_col].replace({0: np.nan})
    # filtrar ou marcar linhas com pouca matrícula
    df.loc[df[mat_col] < min_mat, out_col] = np.nan
    return df

def safe_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """Envoltório para train_test_split com tratamento de stratify opcional."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

# ---------------------------
# 2) Pré-processamento
# ---------------------------

def build_preprocessor(df, numeric_features=None, categorical_features=None):
    """
    Constrói um ColumnTransformer para imputação, encoding e scaling.
    - numeric_features: lista de colunas numéricas
    - categorical_features: lista de colunas categóricas
    Retorna o transformer pronto para usar em Pipeline.
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # remover coluna alvo se presente nas lists (usuário deve garantir)
    # numeric_features = [c for c in numeric_features if c != 'taxa_evasao']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # usar OneHotEncoder com handle_unknown='ignore' para evitar erros quando testar dados novos
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')  # drop outras colunas
    return preprocessor, numeric_features, categorical_features

# ---------------------------
# 3) Modelagem
# ---------------------------

def train_regressors(X_train, y_train, preprocessor, random_state=42, do_grid=False):
    """
    Treina um conjunto de modelos de regressão:
    - RandomForestRegressor
    - MLPRegressor
    Retorna dicionário {name: pipeline}
    Se do_grid=True, efetua busca de hiperparâmetros (GridSearch em parâmetros básicos).
    """
    pipelines = {}

    # Random Forest Regressor
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_pipeline = Pipeline(steps=[('preproc', preprocessor), ('rf', rf)])
    if do_grid:
        param_grid = {
            'rf__n_estimators': [100, 250],
            'rf__max_depth': [10, 20, None],
            'rf__min_samples_leaf': [1, 4]
        }
        gs = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        pipelines['rf'] = gs.best_estimator_
        print("RF best params:", gs.best_params_)
    else:
        rf_pipeline.fit(X_train, y_train)
        pipelines['rf'] = rf_pipeline

    # MLP Regressor
    mlp = MLPRegressor(random_state=random_state, max_iter=500)
    mlp_pipeline = Pipeline(steps=[('preproc', preprocessor), ('mlp', mlp)])
    if do_grid:
        param_grid = {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50,50)],
            'mlp__alpha': [0.0001, 0.001],
            'mlp__learning_rate_init': [0.001, 0.01]
        }
        gs = GridSearchCV(mlp_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        pipelines['mlp'] = gs.best_estimator_
        print("MLP best params:", gs.best_params_)
    else:
        mlp_pipeline.fit(X_train, y_train)
        pipelines['mlp'] = mlp_pipeline

    return pipelines

def evaluate_regression_models(models_dict, X_test, y_test):
    """
    Avalia modelos de regressão e imprime métricas MAE, RMSE e R2.
    Retorna DataFrame resumo.
    """
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
    """
    Converte taxa contínua em rótulo binário:
      - threshold: se None, usa mediana (ou outro quantil desejado)
    Retorna série/binary column.
    """
    if threshold is None:
        threshold = df[rate_col].median(skipna=True)
    return (df[rate_col] > threshold).astype(int), threshold

def train_classifiers(X_train, y_train, preprocessor, random_state=42, do_grid=False):
    """
    Treina:
     - LogisticRegression (baseline)
     - RandomForestClassifier
     - MLPClassifier
    Retorna dicionário de pipelines.
    """
    pipelines = {}

    # Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_pipe = Pipeline(steps=[('preproc', preprocessor), ('lr', lr)])
    if do_grid:
        param_grid = {'lr__C': [0.01, 0.1, 1, 10]}
        gs = GridSearchCV(lr_pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        pipelines['lr'] = gs.best_estimator_
        print("LR best params:", gs.best_params_)
    else:
        lr_pipe.fit(X_train, y_train)
        pipelines['lr'] = lr_pipe

    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rfc_pipe = Pipeline(steps=[('preproc', preprocessor), ('rfc', rfc)])
    if do_grid:
        param_grid = {'rfc__n_estimators': [100, 250], 'rfc__max_depth': [10, None]}
        gs = GridSearchCV(rfc_pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        pipelines['rfc'] = gs.best_estimator_
        print("RFC best params:", gs.best_params_)
    else:
        rfc_pipe.fit(X_train, y_train)
        pipelines['rfc'] = rfc_pipe

    # MLP Classifier
    mlp = MLPClassifier(max_iter=500, random_state=random_state)
    mlp_pipe = Pipeline(steps=[('preproc', preprocessor), ('mlp', mlp)])
    if do_grid:
        param_grid = {'mlp__hidden_layer_sizes': [(50,), (100,)], 'mlp__alpha': [0.0001, 0.001]}
        gs = GridSearchCV(mlp_pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        pipelines['mlp_clf'] = gs.best_estimator_
        print("MLP clf best params:", gs.best_params_)
    else:
        mlp_pipe.fit(X_train, y_train)
        pipelines['mlp_clf'] = mlp_pipe

    return pipelines

def evaluate_classification_models(models_dict, X_test, y_test):
    """
    Avalia modelos de classificação e imprime accuracy, precision, recall, f1 e roc_auc (se disponível).
    Retorna DataFrame resumo.
    """
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
        # tentar probabilidade para ROC AUC
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            scores['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            scores['roc_auc'] = np.nan
        print(f"[{name}] acc={scores['accuracy']:.4f} prec={scores['precision']:.4f} rec={scores['recall']:.4f} f1={scores['f1']:.4f} roc_auc={scores['roc_auc']}")
        rows.append(scores)
    return pd.DataFrame(rows)

# ---------------------------
# 5) Importância de variáveis (após RF)
# ---------------------------

def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    """
    Recupera nomes das features expandidas após ColumnTransformer + OneHot.
    Atenção: depende de OneHotEncoder usando sparse=False (como configurado).
    """
    cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = list(cat_ohe.get_feature_names_out(categorical_features))
    return numeric_features + cat_names

def extract_rf_feature_importances(rf_pipeline, numeric_features, categorical_features, top_n=30):
    """
    Recebe um pipeline cujo último passo é RandomForest e o primeiro é preprocessor.
    Retorna DataFrame com importâncias ordenadas.
    """
    preproc = rf_pipeline.named_steps['preproc']
    rf = rf_pipeline.named_steps[list(rf_pipeline.named_steps.keys())[-1]]
    feature_names = get_feature_names_from_preprocessor(preproc, numeric_features, categorical_features)
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return fi_df.head(top_n)

# ---------------------------
# 6) Exemplo de uso do pipeline
# ---------------------------

if __name__ == "__main__":
    # Ajuste os caminhos para os seus arquivos CSV do INEP (2021,2022,2023, etc.)
    csv_paths = [
        "L_T_MICRODADOS_CADASTRO_CURSOS_2023.csv"
        # "microdados_censo_2022.csv",
        # "microdados_censo_2023.csv",
    ]

    # Leitura e concatenação simples (se esquema igual)
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            print("Lendo", p)
            dfs.append(read_inep_csv(p))
        else:
            print(f"Aviso: arquivo {p} não encontrado. Ajuste o caminho.")
    if len(dfs) == 0:
        raise SystemExit("Nenhum CSV lido. Por favor, coloque os microdados no caminho correto e rode novamente.")
    df = pd.concat(dfs, ignore_index=True)

    # Filtrar cursos de Letras (heurística)
    df = filter_letras(df, course_col='NO_CURSO')

    # Criar target taxa de evasão
    df = compute_taxa_evasao(df,
                             desv_col='QT_SIT_DESVINCULADO',
                             trans_col='QT_SIT_TRANSFERIDO',
                             mat_col='QT_MAT',
                             out_col='taxa_evasao',
                             min_mat=5)  # exemplo: exigir pelo menos 5 matriculados

    # Remover linhas sem target
    df = df.dropna(subset=['taxa_evasao']).reset_index(drop=True)
    print("Registros após filtragem e target criado:", len(df))

    # Seleção inicial de features (ajuste conforme disponibilidade)
   # Incluir
   # QT_VG_TOTAL	QT_VG_TOTAL_DIURNO	QT_VG_TOTAL_NOTURNO	QT_VG_TOTAL_EAD	QT_VG_NOVA	QT_VG_PROC_SELETIVO	QT_VG_REMANESC	QT_VG_PROG_ESPECIAL	QT_INSCRITO_TOTAL	QT_INSCRITO_TOTAL_DIURNO	QT_INSCRITO_TOTAL_NOTURNO	QT_INSCRITO_TOTAL_EAD	QT_INSC_VG_NOVA	QT_INSC_PROC_SELETIVO	QT_INSC_VG_REMANESC	QT_INSC_VG_PROG_ESPECIAL	QT_ING	QT_ING_FEM	QT_ING_MASC	QT_ING_DIURNO	QT_ING_NOTURNO	QT_ING_VG_NOVA	QT_ING_VESTIBULAR	QT_ING_ENEM	QT_ING_AVALIACAO_SERIADA	QT_ING_SELECAO_SIMPLIFICA	QT_ING_EGR	QT_ING_OUTRO_TIPO_SELECAO	QT_ING_PROC_SELETIVO	QT_ING_VG_REMANESC	QT_ING_VG_PROG_ESPECIAL	QT_ING_OUTRA_FORMA	QT_ING_0_17	QT_ING_18_24	QT_ING_25_29	QT_ING_30_34	QT_ING_35_39	QT_ING_40_49	QT_ING_50_59	QT_ING_60_MAIS	QT_ING_BRANCA	QT_ING_PRETA	QT_ING_PARDA	QT_ING_AMARELA	QT_ING_INDIGENA	QT_ING_CORND	QT_MAT	QT_MAT_FEM	QT_MAT_MASC	QT_MAT_DIURNO	QT_MAT_NOTURNO	QT_MAT_0_17	QT_MAT_18_24	QT_MAT_25_29	QT_MAT_30_34	QT_MAT_35_39	QT_MAT_40_49	QT_MAT_50_59	QT_MAT_60_MAIS	QT_MAT_BRANCA	QT_MAT_PRETA	QT_MAT_PARDA	QT_MAT_AMARELA	QT_MAT_INDIGENA	QT_MAT_CORND	QT_CONC	QT_CONC_FEM	QT_CONC_MASC	QT_CONC_DIURNO	QT_CONC_NOTURNO	QT_CONC_0_17	QT_CONC_18_24	QT_CONC_25_29	QT_CONC_30_34	QT_CONC_35_39	QT_CONC_40_49	QT_CONC_50_59	QT_CONC_60_MAIS	QT_CONC_BRANCA	QT_CONC_PRETA	QT_CONC_PARDA	QT_CONC_AMARELA	QT_CONC_INDIGENA	QT_CONC_CORND	QT_ING_NACBRAS	QT_ING_NACESTRANG	QT_MAT_NACBRAS	QT_MAT_NACESTRANG	QT_CONC_NACBRAS	QT_CONC_NACESTRANG	QT_ALUNO_DEFICIENTE	QT_ING_DEFICIENTE	QT_MAT_DEFICIENTE	QT_CONC_DEFICIENTE	QT_ING_FINANC	QT_ING_FINANC_REEMB	QT_ING_FIES	QT_ING_RPFIES	QT_ING_FINANC_REEMB_OUTROS	QT_ING_FINANC_NREEMB	QT_ING_PROUNII	QT_ING_PROUNIP	QT_ING_NRPFIES	QT_ING_FINANC_NREEMB_OUTROS	QT_MAT_FINANC	QT_MAT_FINANC_REEMB	QT_MAT_FIES	QT_MAT_RPFIES	QT_MAT_FINANC_REEMB_OUTROS	QT_MAT_FINANC_NREEMB	QT_MAT_PROUNII	QT_MAT_PROUNIP	QT_MAT_NRPFIES	QT_MAT_FINANC_NREEMB_OUTROS	QT_CONC_FINANC	QT_CONC_FINANC_REEMB	QT_CONC_FIES	QT_CONC_RPFIES	QT_CONC_FINANC_REEMB_OUTROS	QT_CONC_FINANC_NREEMB	QT_CONC_PROUNII	QT_CONC_PROUNIP	QT_CONC_NRPFIES	QT_CONC_FINANC_NREEMB_OUTROS	QT_ING_RESERVA_VAGA	QT_ING_RVREDEPUBLICA	QT_ING_RVETNICO	QT_ING_RVPDEF	QT_ING_RVSOCIAL_RF	QT_ING_RVOUTROS	QT_MAT_RESERVA_VAGA	QT_MAT_RVREDEPUBLICA	QT_MAT_RVETNICO	QT_MAT_RVPDEF	QT_MAT_RVSOCIAL_RF	QT_MAT_RVOUTROS	QT_CONC_RESERVA_VAGA	QT_CONC_RVREDEPUBLICA	QT_CONC_RVETNICO	QT_CONC_RVPDEF	QT_CONC_RVSOCIAL_RF	QT_CONC_RVOUTROS	QT_SIT_TRANCADA	QT_SIT_DESVINCULADO	QT_SIT_TRANSFERIDO	QT_SIT_FALECIDO	QT_ING_PROCESCPUBLICA	QT_ING_PROCESCPRIVADA	QT_ING_PROCNAOINFORMADA	QT_MAT_PROCESCPUBLICA	QT_MAT_PROCESCPRIVADA	QT_MAT_PROCNAOINFORMADA	QT_CONC_PROCESCPUBLICA	QT_CONC_PROCESCPRIVADA	QT_CONC_PROCNAOINFORMADA	QT_PARFOR	QT_ING_PARFOR	QT_MAT_PARFOR	QT_CONC_PARFOR	QT_APOIO_SOCIAL	QT_ING_APOIO_SOCIAL	QT_MAT_APOIO_SOCIAL	QT_CONC_APOIO_SOCIAL	QT_ATIV_EXTRACURRICULAR	QT_ING_ATIV_EXTRACURRICULAR	QT_MAT_ATIV_EXTRACURRICULAR	QT_CONC_ATIV_EXTRACURRICULAR	QT_MOB_ACADEMICA	QT_ING_MOB_ACADEMICA	QT_MAT_MOB_ACADEMICA	QT_CONC_MOB_ACADEMICA
    candidate_numeric = [
      'QT_MAT',
        'QT_APOIO_SOCIAL', 'QT_ATIV_EXTRACURRICULAR'
    ]
    # garantir que existam no dataframe
    numeric_features = [c for c in candidate_numeric if c in df.columns]
    categorical_features = [c for c in [
        'CO_REGIAO', 'CO_UF', 'CO_MUNICIPIO', 'IN_CAPITAL',
        'TP_REDE', 'TP_CATEGORIA_ADMINISTRATIVA', 'TP_ORGANIZACAO_ACADEMICA',
        'TP_MODALIDADE_ENSINO', 'TP_GRAU_ACADEMICO'
    ] if c in df.columns]

    # Criar preprocessor
    preprocessor, numeric_features, categorical_features = build_preprocessor(df,
                                                                             numeric_features=numeric_features,
                                                                             categorical_features=categorical_features)

    # Separar X e y (regressão)
    X = df[numeric_features + categorical_features].copy()
    y_reg = df['taxa_evasao'].astype(float).copy()

    # Train/test split (regressão)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Treinar regressors
    print("Treinando modelos de regressão...")
    models_reg = train_regressors(X_train, y_train, preprocessor, do_grid=False)

    # Avaliar regressors
    print("Avaliando regressão...")
    reg_res = evaluate_regression_models(models_reg, X_test, y_test)
    print(reg_res)

    # Extrair importâncias de RF (se disponível)
    if 'rf' in models_reg:
        try:
            fi = extract_rf_feature_importances(models_reg['rf'], numeric_features, categorical_features, top_n=40)
            print("Top importâncias (Random Forest):")
            print(fi.head(20).to_string(index=False))
            # plot simples
            plt.figure(figsize=(8,6))
            sns.barplot(data=fi.head(20), x='importance', y='feature')
            plt.title('Top 20 importâncias - Random Forest (regression)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Não foi possível extrair importâncias:", e)

    # -------------------------
    # Classificação opcional
    # -------------------------
    # transformar taxa em binária (alto risco / baixo risco) usando mediana
    y_bin, thresh = make_binary_target_from_rate(df, rate_col='taxa_evasao', threshold=None)
    print("Threshold binário (mediana):", thresh)

    Xc = df[numeric_features + categorical_features].copy()
    yc = y_bin

    # split com stratify
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

    print("Treinando classificadores...")
    models_clf = train_classifiers(Xc_train, yc_train, preprocessor, do_grid=False)

    print("Avaliando classificadores...")
    clf_res = evaluate_classification_models(models_clf, Xc_test, yc_test)
    print(clf_res)

    # salvar modelos (exemplo)
    os.makedirs('models', exist_ok=True)
    for name, mdl in {**models_reg, **models_clf}.items():
        fname = os.path.join('models', f'{name}.joblib')
        try:
            joblib.dump(mdl, fname)
            print("Salvo:", fname)
        except Exception as e:
            print("Falha ao salvar", name, e)

    print("Pipeline concluído.")
