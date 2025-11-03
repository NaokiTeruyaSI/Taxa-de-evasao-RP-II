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
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Configurações de Diretório Otimizadas ---
# O novo diretório de saída será 'microdados_otimizados'
INPUT_DIR = 'microdados'
OUTPUT_DIR_ORIGINAL = 'microdados_filtrados' # Diretório onde os arquivos COMPLETO_LETRAS_LIC_*.CSV estão
OUTPUT_DIR_OTIMIZADO = 'microdados_otimizados'
OUTPUT_DIR_ANALISE_OTIMIZADA = os.path.join(OUTPUT_DIR_OTIMIZADO, 'tabelas_analise') 

# Cria as pastas de saída otimizadas se elas não existirem
Path(OUTPUT_DIR_OTIMIZADO).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE_OTIMIZADA).mkdir(parents=True, exist_ok=True)

# --- Definição do Conjunto de Features Otimizadas (33 Colunas) ---
# Selecionadas pela Importância Média (2022-2024) do Classificador
COLUNAS_OTIMIZADAS: List[str] = [
    'CO_MUNICIPIO', 'CO_CURSO', 'QT_SIT_TRANCADA', 'QT_MAT', 'QT_MAT_NACBRAS', 
    'CO_UF', 'CO_CINE_ROTULO', 'QT_MAT_PROCESCPUBLICA', 'QT_MAT_FEM', 'QT_MAT_BRANCA',
    'CO_REGIAO', 'QT_ING_PROCESCPUBLICA', 'TP_ORGANIZACAO_ACADEMICA', 'QT_ING_FEM', 
    'QT_ING_VESTIBULAR', 'QT_ING_BRANCA', 'QT_ING_VG_NOVA', 'QT_MAT_MASC', 
    'QT_MAT_PARDA', 'QT_MAT_18_24', 'QT_ING', 'QT_ING_NACBRAS', 'QT_ING_18_24', 
    'QT_ING_PARDA', 'QT_MAT_FINANC', 'QT_ING_MASC', 'QT_MAT_25_29', 'QT_MAT_40_49', 
    'QT_MAT_CORND', 'QT_MAT_FINANC_NREEMB', 
    'TX_EVASAO' 
]

# ======================================================================
# 🔍 Função para extrair importâncias de features do modelo Random Forest (Mantida)
# ======================================================================

def get_feature_importances(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extrai e mapeia a importância das features de um modelo Random Forest
    dentro de um Pipeline com ColumnTransformer e OrdinalEncoder.
    """
    try:
        model = pipeline.named_steps.get('model')
        preprocessor = pipeline.named_steps.get('preprocess')

        if model is None or preprocessor is None:
            print("⚠️ Estrutura inesperada no Pipeline (faltam etapas).")
            return pd.DataFrame()

        if not hasattr(model, "feature_importances_"):
            print("⚠️ Modelo não possui 'feature_importances_' (não é RandomForest).")
            return pd.DataFrame()

        importances = model.feature_importances_

        # A lógica de reconstrução de nomes é mantida
        output_features = []
        for name, transformer, cols in preprocessor.transformers_:
            if transformer == 'drop':
                continue
            elif name == 'num':
                output_features.extend(cols)
            elif name == 'cat':
                output_features.extend(cols)
            elif name == 'remainder' and transformer == 'passthrough':
                remainder_cols = [f for f in feature_names if f not in output_features]
                output_features.extend(remainder_cols)

        feature_importance_df = pd.DataFrame({
            'Feature': output_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        feature_importance_df['Importance'] = feature_importance_df['Importance'].round(4)
        return feature_importance_df

    except Exception as e:
        print(f"⚠️ Erro ao extrair feature importances: {e}")
        return pd.DataFrame()


# =======================================================
# 🤖 Treinamento de Modelos: RandomForest e MLP Regressor (Mantida)
# =======================================================

def treinar_preditor_evasao(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Treina RandomForest e MLP para prever TX_EVASAO e calcula Feature Importance.
    """

    if 'TX_EVASAO' not in df.columns:
        raise ValueError("Coluna 'TX_EVASAO' não encontrada no DataFrame.")

    # Alvo e features
    y = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
    X = df.drop(columns=['TX_EVASAO']).copy()

    # Identifica tipos de colunas
    colunas_numericas = [c for c in X.columns if c.startswith('QT_')]
    colunas_categoricas = [c for c in X.columns if not c.startswith('QT_')]

    # Limpa colunas categóricas (remove aspas, espaços e converte para string)
    for c in colunas_categoricas:
        X[c] = X[c].astype(str).str.replace('"', '').str.strip()

    # Pré-processamento
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), colunas_numericas),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), colunas_categoricas)
    ])

    # Split de treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos
    modelos = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    }

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"  ⚙️ Treinando modelo: {nome}")
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', modelo)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        resultados[nome] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'modelo': pipe
        }

        # Importância do Random Forest
        if nome == 'RandomForest':
            all_features = colunas_numericas + colunas_categoricas
            importance_df = get_feature_importances(pipe, all_features)
            resultados[nome]['FeatureImportance'] = importance_df

            if not importance_df.empty:
                print("\n  ⭐ Top 5 Features Mais Importantes (Random Forest):")
                print(importance_df.head(5).to_markdown(index=False))

    return resultados


# =======================================================
# 📊 Avaliação dos Modelos em Todos os CSVs Processados (MODIFICADA)
# =======================================================

def avaliar_modelos_em_tabelas(output_dir_original: str, colunas_otimizadas: List[str]):
    """
    Roda o preditor apenas nas colunas otimizadas, lendo os arquivos COMPLETO.
    """
    # Busca apenas pelos arquivos COMPLETO, que contêm todas as features
    csv_paths = glob.glob(os.path.join(output_dir_original, 'COMPLETO_LETRAS_LIC_*.CSV'))

    resultados_gerais = []
    importances_gerais = {}

    if not csv_paths:
        print(f"Nenhum CSV COMPLETO encontrado na pasta '{output_dir_original}'.")
        return pd.DataFrame(), {}

    for csv_path in csv_paths:
        nome_completo = os.path.basename(csv_path)
        nome_otimizado = 'OTIMIZADO_' + nome_completo.replace('COMPLETO_', '')
        print(f"\n📊 Avaliando modelo (OTIMIZADO) em: {nome_completo}")

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1')

            # 🛑 FILTRAGEM PELAS COLUNAS OTIMIZADAS AQUI 🛑
            cols_to_keep = [col for col in colunas_otimizadas if col in df.columns]
            
            # Garante que as colunas de cálculo estejam presentes
            if 'TX_EVASAO' in df.columns:
                df_filtrado = df[cols_to_keep].copy()
            else:
                print(f"  ❌ Coluna 'TX_EVASAO' ausente. Pulando {nome_completo}.")
                continue
            
            # Dropna após a seleção para garantir a limpeza do conjunto reduzido
            df_filtrado = df_filtrado.dropna().copy()

            if df_filtrado.shape[0] < 50:
                print(f"  ⚠️ Poucos registros ({df_filtrado.shape[0]}) após filtragem. Pulando...")
                continue
            
            print(f"  -> Dataset Otimizado com {df_filtrado.shape[0]} linhas e {df_filtrado.shape[1]} colunas.")

            resultados = treinar_preditor_evasao(df_filtrado)

            for modelo, metricas in resultados.items():
                resultados_gerais.append({
                    'arquivo': nome_otimizado,
                    'modelo': modelo,
                    'R2': metricas['R2'],
                    'MAE': metricas['MAE'],
                    'RMSE': metricas['RMSE']
                })

                if modelo == 'RandomForest' and 'FeatureImportance' in metricas:
                    importances_gerais[nome_otimizado] = metricas['FeatureImportance']

        except Exception as e:
            print(f"❌ Erro ao processar {nome_completo}: {e}")

    df_resultados = pd.DataFrame(resultados_gerais)

    if not df_resultados.empty:
        print("\n✅ Comparativo final de desempenho (OTIMIZADO):\n")
        print(df_resultados.sort_values(by=['arquivo', 'R2'], ascending=[True, False]).to_markdown(index=False))

    return df_resultados, importances_gerais


# =======================================================
# 🚀 Execução Principal
# =======================================================

if __name__ == '__main__':
    df_resultados, importances_finais = avaliar_modelos_em_tabelas(
        OUTPUT_DIR_ORIGINAL,
        COLUNAS_OTIMIZADAS
    )

    # Salva métricas
    if not df_resultados.empty:
        resultados_path = os.path.join(OUTPUT_DIR_OTIMIZADO, 'comparativo_modelos_regressao_otimizado.csv')
        df_resultados.to_csv(resultados_path, index=False, sep=';', encoding='utf-8')
        print(f"\n📁 Resultados de métricas OTIMIZADAS salvos em: {resultados_path}")

    # Salva importâncias
    for filename, df_imp in importances_finais.items():
        imp_path = os.path.join(OUTPUT_DIR_OTIMIZADO, f'IMPORTANCE_RF_REGRESSAO_{filename}.csv')
        df_imp.to_csv(imp_path, index=False, sep=';', encoding='utf-8')
        print(f"📁 Importância de features OTIMIZADA salva: {imp_path}")