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

# --- Configurações de Diretório ---
INPUT_DIR = 'microdados'
OUTPUT_DIR = 'microdados_filtrados'
OUTPUT_DIR_ANALISE = os.path.join(OUTPUT_DIR, 'tabelas_analise')

# Cria pastas, se necessário
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE).mkdir(parents=True, exist_ok=True)


# ======================================================================
# 🔍 Função para extrair importâncias de features do modelo Random Forest
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

        # Reconstrói nomes das features do ColumnTransformer
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
# 🤖 Treinamento de Modelos: RandomForest e MLP Regressor
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
# 📊 Avaliação dos Modelos em Todos os CSVs Processados
# =======================================================

def avaliar_modelos_em_tabelas(output_dir: str, output_dir_analise: str):
    """
    Roda o preditor em todas as tabelas completas e de análise e compara resultados.
    """
    csv_paths = (
        glob.glob(os.path.join(output_dir, 'COMPLETO_LETRAS_LIC_*.CSV')) +
        glob.glob(os.path.join(output_dir_analise, 'ANALISE_EVASAO_*.CSV'))
    )

    resultados_gerais = []
    importances_gerais = {}

    if not csv_paths:
        print("Nenhum CSV encontrado para análise.")
        return pd.DataFrame(), {}

    for csv_path in csv_paths:
        nome = os.path.basename(csv_path)
        print(f"\n📊 Avaliando modelo em: {nome}")

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1')

            df['TX_EVASAO'] = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
            df = df.dropna(subset=['TX_EVASAO'])

            if df.shape[0] < 50:
                print(f"  ⚠️ Poucos registros ({df.shape[0]}). Pulando...")
                continue

            resultados = treinar_preditor_evasao(df)

            for modelo, metricas in resultados.items():
                resultados_gerais.append({
                    'arquivo': nome,
                    'modelo': modelo,
                    'R2': metricas['R2'],
                    'MAE': metricas['MAE'],
                    'RMSE': metricas['RMSE']
                })

                if modelo == 'RandomForest' and 'FeatureImportance' in metricas:
                    importances_gerais[nome] = metricas['FeatureImportance']

        except Exception as e:
            print(f"❌ Erro ao processar {nome}: {e}")

    df_resultados = pd.DataFrame(resultados_gerais)

    if not df_resultados.empty:
        print("\n✅ Comparativo final de desempenho:\n")
        print(df_resultados.sort_values(by=['arquivo', 'R2'], ascending=[True, False]).to_markdown(index=False))

    return df_resultados, importances_gerais


# =======================================================
# 🚀 Execução Principal
# =======================================================

if __name__ == '__main__':
    df_resultados, importances_finais = avaliar_modelos_em_tabelas(
        OUTPUT_DIR,
        OUTPUT_DIR_ANALISE
    )

    # Salva métricas
    if not df_resultados.empty:
        resultados_path = os.path.join(OUTPUT_DIR, 'comparativo_modelos_evasao.csv')
        df_resultados.to_csv(resultados_path, index=False, sep=';', encoding='utf-8')
        print(f"\n📁 Resultados de métricas salvos em: {resultados_path}")

    # Salva importâncias
    for filename, df_imp in importances_finais.items():
        imp_path = os.path.join(OUTPUT_DIR, f'IMPORTANCE_RF_{filename}.csv')
        df_imp.to_csv(imp_path, index=False, sep=';', encoding='utf-8')
        print(f"📁 Importância de features salva: {imp_path}")
