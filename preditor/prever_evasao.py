# -*- coding: utf-8 -*-
"""
Script para prever a taxa de evasão de um curso específico.

Este programa carrega os modelos e artefatos treinados pelo 
'treinamento_pipeline.py' para fazer uma previsão com base no
código da IES e no ano fornecidos pelo usuário.
"""

import os
import argparse
import joblib
import pandas as pd

def read_inep_csv(path, sep=';', encoding='latin1'):
    """Lê um arquivo CSV do INEP."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {path}")
    return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)

def predict_evasao(codigo_ies, ano, dados_path, artifacts_path='models/prediction_artifacts.joblib'):
    """
    Carrega modelos, encontra os dados da IES/ano e realiza a previsão.
    """
    # 1. Carregar os artefatos de treinamento
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Arquivo de artefatos '{artifacts_path}' não encontrado. "
                                "Execute o script 'treinamento_pipeline.py' primeiro.")
    
    artifacts = joblib.load(artifacts_path)
    reg_models = artifacts['reg_models']
    clf_models = artifacts['clf_models']
    features = artifacts['features']
    threshold = artifacts['classification_threshold']
    
    print(f"Artefatos carregados. Modelos de regressão: {list(reg_models.keys())}")
    print(f"Modelos de classificação: {list(clf_models.keys())}")
    print("-" * 30)

    # 2. Carregar os dados e encontrar o registro específico
    # Usando o mesmo filtro de 'Letras' do treinamento
    df_full = read_inep_csv(dados_path)
    mask_letras = df_full['NO_CURSO'].astype(str).str.upper().str.contains('LETRAS')
    
    # Filtrar pelo código da IES, ano e curso de Letras
    record = df_full[
        (df_full['CO_IES'] == codigo_ies) &
        (df_full['NU_ANO_CENSO'] == ano) &
        (mask_letras)
    ]

    if record.empty:
        print(f"ERRO: Nenhum curso de 'Letras' encontrado para a IES com código {codigo_ies} no ano {ano}.")
        return

    # Se houver mais de um curso de Letras, usamos o primeiro encontrado
    if len(record) > 1:
        print(f"Aviso: Múltiplos ({len(record)}) cursos de Letras encontrados para a IES {codigo_ies} no ano {ano}. Usando o primeiro registro.")
        record = record.head(1)
        
    print(f"Registro encontrado para a IES {codigo_ies} | Curso: {record['NO_CURSO'].iloc[0]}")
    print(f"Modalidade: {record['TP_MODALIDADE_ENSINO'].iloc[0]}")
    print("-" * 30)

    # 3. Preparar os dados para a previsão
    X_pred = record[features]

    # 4. Realizar e exibir as previsões
    print("Resultados da Previsão:\n")

    # Regressão: Previsão da taxa exata
    print("--- Modelos de Regressão (Taxa de Evasão) ---")
    for name, model in reg_models.items():
        prediction = model.predict(X_pred)[0]
        print(f"Modelo '{name}': Taxa de evasão prevista = {prediction:.2%}")

    print("\n--- Modelos de Classificação (Risco de Alta Evasão) ---")
    print(f"(Limiar para 'alta evasão' é > {threshold:.2%})")
    # Classificação: Previsão de risco (taxa acima da mediana)
    for name, model in clf_models.items():
        proba = model.predict_proba(X_pred)[0][1] # Probabilidade da classe 1 (alta evasão)
        pred_class = model.predict(X_pred)[0]
        risco = "Alto" if pred_class == 1 else "Baixo"
        print(f"Modelo '{name}': Probabilidade de alta evasão = {proba:.2%} (Risco: {risco})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prever a taxa de evasão para uma IES e ano específicos.")
    parser.add_argument("--codigo_ies", type=int, required=True, help="Código da Instituição de Ensino Superior (IES).")
    parser.add_argument("--ano", type=int, required=True, help="Ano do censo a ser utilizado para a previsão.")
    parser.add_argument("--dados_path", type=str, required=True, help="Caminho para o arquivo CSV de microdados do INEP.")
    parser.add_argument("--artifacts_path", type=str, default="models/prediction_artifacts.joblib", help="Caminho para o arquivo de artefatos do modelo treinado.")
    
    args = parser.parse_args()
    
    predict_evasao(
        codigo_ies=args.codigo_ies,
        ano=args.ano,
        dados_path=args.dados_path,
        artifacts_path=args.artifacts_path
    )