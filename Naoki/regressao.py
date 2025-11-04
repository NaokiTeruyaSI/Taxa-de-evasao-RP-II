import pandas as pd
import os
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer 
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import traceback # Adicionado para debug

# --- NOVOS IMPORTS ---
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# ---------------------

# --- Configurações de Diretório ---
INPUT_DIR = 'microdados'
OUTPUT_DIR = 'microdados_filtrados_simples'
OUTPUT_DIR_ANALISE = os.path.join(OUTPUT_DIR, 'tabelas_analise_simples')

# Cria pastas, se necessário
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE).mkdir(parents=True, exist_ok=True)


# ======================================================================
# 🔍 Função para extrair importâncias de features (ATUALIZADA)
# ======================================================================

def get_feature_importances(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extrai e mapeia a importância das features de modelos baseados em árvore.
    """
    try:
        model = pipeline.named_steps['model']
        
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ Modelo '{type(model).__name__}' não possui 'feature_importances_'.")
            return pd.DataFrame()
            
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names, # Usa a lista de features já ordenada
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        feature_importance_df['Importance'] = feature_importance_df['Importance'].round(4)
        return feature_importance_df

    except Exception as e:
        print(f"⚠️ Erro ao extrair feature importances: {e}")
        return pd.DataFrame()


# =======================================================
# 🤖 Treinamento de Modelos (ATUALIZADO)
# =======================================================

def treinar_preditor_evasao(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Treina múltiplos regressores para prever np.log1p(TX_EVASAO) 
    e calcula métricas na escala original.
    """

    if 'TX_EVASAO' not in df.columns:
        raise ValueError("Coluna 'TX_EVASAO' não encontrada no DataFrame.")

    # Alvo e features
    y_raw = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
    X = df.drop(columns=['TX_EVASAO']).copy()
    
    # --- 2. TRANSFORMAÇÃO DO ALVO (Y) ---
    y = np.log1p(y_raw)
    # -----------------------------------

    # Identifica tipos de colunas
    colunas_numericas = []
    colunas_categoricas = []
    feat_categoricas_conhecidas = ['FEAT_DOMINANCIA_MODALIDADE'] # Do seu script de classificação
    ignorar = ['CO_MUNICIPIO', 'CO_CURSO', 'CO_IES'] # Do seu script de classificação

    # Lógica de classificação de colunas melhorada
    print("⚙️ Classificando variáveis (numéricas x categóricas)...")
    for c in X.columns:
        if c in ignorar:
            continue
        if c.startswith(('QT_', 'FEAT_')) and c not in feat_categoricas_conhecidas:
            colunas_numericas.append(c)
        else:
            colunas_categoricas.append(c)
            
    print(f"   ➤ Numéricas: {len(colunas_numericas)} | Categóricas: {len(colunas_categoricas)}")
    all_features = colunas_numericas + colunas_categoricas # Ordem correta

    # Limpa colunas categóricas
    for c in colunas_categoricas:
        X[c] = X[c].astype(str).str.replace('"', '').str.strip()

    # --- 3. PIPELINES DE PRÉ-PROCESSAMENTO ---
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, colunas_numericas), 
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), colunas_categoricas)
    ])
    # ----------------------------------------

    # Split de treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- DICIONÁRIO DE MODELOS ATUALIZADO ---
    modelos = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300, 
            random_state=42, 
            n_jobs=-1,
            max_depth=15,          # Adicionado (baseado no script anterior)
            min_samples_leaf=5     # Adicionado (baseado no script anterior)
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(128, 64), 
            activation='tanh',     # 'tanh' ou 'relu' são boas escolhas
            solver='adam', 
            max_iter=5000, 
            random_state=42,
            alpha=0.001
        ),
        # --- NOVOS MODELOS ADICIONADOS ---
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            # class_weight='balanced' REMOVIDO (apenas classificação)
            random_state=42,
            n_jobs=-1,
            verbose=-1 # Para suprimir avisos
        )
    }
    # -----------------------------------------

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"  ⚙️ Treinando modelo: {nome}")
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', modelo)
        ])

        try:
            pipe.fit(X_train, y_train)
            y_pred_log = pipe.predict(X_test) # Previsões estão em escala de log

            # --- REVERSÃO DAS MÉTRICAS ---
            y_test_original = np.expm1(y_test)
            y_pred_original = np.expm1(y_pred_log)
            y_pred_original[y_pred_original < 0] = 0 

            # Métricas
            mae = mean_absolute_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            r2 = r2_score(y_test_original, y_pred_original)
            # --------------------------------

            resultados[nome] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'modelo': pipe
            }

            # --- LÓGICA DE IMPORTÂNCIA ATUALIZADA ---
            # Pega importância de RF, XGB, LGBM
            if hasattr(modelo, "feature_importances_"):
                imp = get_feature_importances(pipe, all_features)
                if imp is not None and not imp.empty:
                    resultados[nome]['FeatureImportance'] = imp
                    print(f"\n⭐ Top 5 Features mais relevantes ({nome}):")
                    print(imp.head(5).to_markdown(index=False))
            # -----------------------------------------
            
        except Exception as e:
            print(f"❌ Erro ao treinar {nome}: {e}")
            traceback.print_exc() # Mostra o erro detalhado
            resultados[nome] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'modelo': None}

    
    print("  -> Treinamento concluído.")
    return resultados


# =======================================================
# 📊 Avaliação (Função Mantida, mas agora usa 'ignorar')
# =======================================================

def avaliar_modelos_em_tabelas(output_dir: str, output_dir_analise: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Roda o preditor em todas as tabelas completas e de análise e compara resultados.
    """
    
    # --- LÓGICA DE BUSCA DE ARQUIVOS ATUALIZADA ---
    # Foca em ler os arquivos do 'filtrar.py' do diretório de saída
    all_files = glob.glob(os.path.join(output_dir, "*.csv"))
    if not all_files:
        print("Nenhum CSV encontrado em 'microdados_filtrados'.")
        return pd.DataFrame(), {}
    
    print(f"\n📂 Encontrados {len(all_files)} arquivos CSV em '{output_dir}':")
    for f in all_files:
        print(f"   - {os.path.basename(f)}")
    # -------------------------------------------

    resultados_gerais = []
    importances_gerais = {}

    for csv_path in all_files:
        nome = os.path.basename(csv_path)
        print(f"\n{'-'*60}")
        print(f"📄 Processando arquivo: {nome}")

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1', low_memory=False)
            print(f"   ➤ Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")

            df['TX_EVASAO'] = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
            df = df.dropna(subset=['TX_EVASAO']) # Remove apenas se o ALVO for nulo

            if df.shape[0] < 50:
                print(f"  ⚠️ Poucos registros ({df.shape[0]}). Pulando...")
                continue
                
            # # --- AVISO DE DATA LEAKAGE (Adicionado por responsabilidade) ---
            # features_com_leakage = [col for col in df.columns if '_MAT' in col or '_CONC' in col]
            # if features_com_leakage:
            #     print("\n" + "!"*60)
            #     print("⚠️ AVISO DE DATA LEAKAGE (VAZAMENTO DE DADOS) ⚠️")
            #     print("  Este arquivo contém features baseadas em 'QT_MAT' e 'QT_CONC'.")
            #     print("  Para um modelo preditivo real, elas devem ser removidas do 'filtrar.py'.")
            #     print("!"*60)
            # # --- FIM DO AVISO ---

            resultados = treinar_preditor_evasao(df)

            for modelo, metricas in resultados.items():
                if metricas['modelo'] is not None: # Só adiciona se o treino foi bem sucedido
                    resultados_gerais.append({
                        'arquivo': nome,
                        'modelo': modelo,
                        'R2': metricas['R2'],
                        'MAE': metricas['MAE'],
                        'RMSE': metricas['RMSE']
                    })

                    if 'FeatureImportance' in metricas and not metricas['FeatureImportance'].empty:
                        imp = metricas['FeatureImportance'].copy()
                        imp['Arquivo'] = nome
                        # Adiciona a um dict para evitar duplicatas (usa o último)
                        importances_gerais[f"{nome}_{modelo}"] = imp

        except Exception as e:
            print(f"❌ Erro ao processar {nome}: {e}")
            traceback.print_exc()

    df_resultados = pd.DataFrame(resultados_gerais)
    return df_resultados, importances_gerais


# =======================================================
# 📈 FUNÇÃO MEGA-DATASET REMOVIDA
# (A função 'avaliar_modelos_em_tabelas' agora já lida
#  com o 'mega_dataset_filtrado.csv' se ele estiver na pasta)
# =======================================================


# =======================================================
# 🚀 Execução Principal (Simplificada)
# =======================================================

if __name__ == '__main__':
    
    print("=" * 60)
    print("📊 INÍCIO DA ANÁLISE DE REGRESSÃO")
    print("=" * 60)
    
    # 1. Roda a análise para TODOS os arquivos em 'microdados_filtrados'
    #    (Isso inclui 'microdados_filtrado_2022.csv', '...2023.csv' E 
    #     'mega_dataset_filtrado.csv', se existirem)
    df_resultados_finais, importances_finais_dict = avaliar_modelos_em_tabelas(
        OUTPUT_DIR,
        OUTPUT_DIR_ANALISE # Esta variável não é mais usada aqui, mas mantida
    )
    
    # 2. Salva os resultados combinados
    if not df_resultados_finais.empty:
        resultados_path = os.path.join(OUTPUT_DIR_ANALISE, 'metricas_REGRESSAO_CONSOLIDADO.csv')
        df_resultados_finais = df_resultados_finais.sort_values(by=['arquivo', 'R2'], ascending=[True, False])
        df_resultados_finais.to_csv(resultados_path, index=False, sep=';', encoding='utf-8')
        
        print(f"\n📁 Resultados de métricas consolidadas salvos em: '{resultados_path}'")
        
        # Imprime o comparativo final completo
        print("\n✅ Comparativo final de desempenho (REGRESSÃO):\n")
        print(df_resultados_finais.to_markdown(index=False, numalign="left", stralign="left"))

    # 3. Salva importâncias
    if importances_finais_dict:
        # Concatena todas as importâncias em um único arquivo
        df_import_final = pd.concat(importances_finais_dict.values(), ignore_index=True)
        imp_path = os.path.join(OUTPUT_DIR_ANALISE, 'IMPORTANCE_FEATURES_REGRESSAO_CONSOLIDADO.csv')
        df_import_final.to_csv(imp_path, index=False, sep=';', encoding='utf-8')
        print(f"\n📁 Importância de Features consolidada salva em: {imp_path}")
    else:
        print("\n⚠️ Nenhuma importância de feature foi gerada.")

    print("\n✅ Análise de Regressão Concluída com sucesso!")
    print("=" * 60)