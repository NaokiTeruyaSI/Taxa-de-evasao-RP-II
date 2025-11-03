import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- Configurações de Diretório (Mantidas) ---
INPUT_DIR = 'microdados'
OUTPUT_DIR = 'microdados_filtrados'
OUTPUT_DIR_ANALISE = os.path.join(OUTPUT_DIR, 'tabelas_analise') 

# Cria as pastas de saída se elas não existirem (Mantidas)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE).mkdir(parents=True, exist_ok=True)


def criar_alvo_binario_alto_risco(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Cria a variável alvo binária ALTO_RISCO (1 se TX_EVASAO >= mediana), 
    calculando a mediana APENAS em valores > 0 E com volume mínimo de ingressantes (QT_ING >= 10).
    Retorna o DataFrame modificado e o valor do limiar (mediana).
    """
    if 'TX_EVASAO' not in df.columns or 'QT_ING' not in df.columns:
        raise ValueError("Colunas 'TX_EVASAO' ou 'QT_ING' ausentes para criar o alvo binário.")
    
    # 1. Converte para numérico e remove NaNs
    df['TX_EVASAO'] = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
    df['QT_ING'] = pd.to_numeric(df['QT_ING'], errors='coerce').fillna(0) # Garante QT_ING numérico
    df_clean = df.dropna(subset=['TX_EVASAO']).copy()
    
    if df_clean.empty:
        return df_clean, 0.0

    # 2. CALCULA A MEDIANA APENAS EM VALORES > 0 E COM VOLUME MÍNIMO
    QT_ING_MINIMO = 1 # <--- LIMITE DE VOLUME IMPLEMENTADO AQUI
    
    evasao_filtrada_para_mediana = df_clean.loc[
        (df_clean['TX_EVASAO'] > 0) & 
        (df_clean['QT_ING'] >= QT_ING_MINIMO),
        'TX_EVASAO'
    ]
    
    if evasao_filtrada_para_mediana.empty:
        # Se não houver cursos com volume mínimo e evasão, o limiar é 0.
        mediana = 0.0
        print(f"  -> Aviso: Nenhum curso com TX_EVASAO > 0 e QT_ING >= {QT_ING_MINIMO}. Limiar de Alto Risco será 0.0.")
    else:
        mediana = evasao_filtrada_para_mediana.median()
        
    # 3. Cria o alvo binário (aplicado a TODOS os registros, independentemente do QT_ING)
    # Apenas o cálculo da MEDIANA usa o filtro de volume.
    df_clean['ALTO_RISCO'] = (df_clean['TX_EVASAO'] >= mediana).astype(int)
    
    # Remove a TX_EVASAO original para que ela não vaze como feature
    df_clean = df_clean.drop(columns=['TX_EVASAO'])
    
    return df_clean, mediana


# --- As demais funções (get_feature_importances, treinar_classificadores_evasao, avaliar_modelos_em_tabelas e main) permanecem as mesmas. ---

def get_feature_importances(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extrai e mapeia a importância das features de um modelo Random Forest Classifier.
    """
    try:
        model = pipeline.named_steps['model']
        
        # 1. Obtém as importâncias diretamente do Random Forest (Classificador)
        importances = model.feature_importances_
        
        # 2. Mapeamento dos Nomes das Features (baseado na ordem do ColumnTransformer)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return feature_importance_df

    except AttributeError as e:
        # Acessa o modelo (RandomForestClassifier)
        if hasattr(pipeline.named_steps.get('model'), 'feature_importances_'):
            pass # Ignora, pois é o caso do MLP sem 'feature_importances_'
        else:
            print(f"⚠️ Aviso: Não foi possível extrair feature importance: {e}")
        return pd.DataFrame()


def treinar_classificadores_evasao(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Treina RandomForest, MLP e LogisticRegression para prever ALTO_RISCO.
    """
    if 'ALTO_RISCO' not in df.columns:
        raise ValueError("Coluna 'ALTO_RISCO' não encontrada no DataFrame.")

    # Separa alvo e features
    y = df['ALTO_RISCO']
    X = df.drop(columns=['ALTO_RISCO']).copy()

    # Identifica colunas
    colunas_numericas = [c for c in X.columns if c.startswith('QT_')]
    colunas_categoricas = [c for c in X.columns if not c.startswith('QT_')]

    # Converter categóricas para string (garante que OrdinalEncoder funcione)
    for c in colunas_categoricas:
        X[c] = X[c].astype(str)

    # Pré-processamento:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), colunas_numericas),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), colunas_categoricas)
    ])
    all_features = colunas_numericas + colunas_categoricas

    # Divide dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Modelos (Classificadores)
    modelos = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=200)
    }

    resultados = {}

    for nome, modelo in modelos.items():
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', modelo)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Para ROC-AUC, precisamos das probabilidades (predict_proba)
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1] # Pega a probabilidade da classe positiva (1)
        except AttributeError:
            y_proba = y_pred # fallback, embora não seja ideal para AUC

        # --- Métricas de Classificação ---
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        resultados[nome] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'ROC-AUC': auc,
            'modelo': pipe
        }
        
        # ANÁLISE DE IMPORTÂNCIA DO RANDOM FOREST
        if nome == 'RandomForest':
            importance_df = get_feature_importances(pipe, all_features)
            resultados[nome]['FeatureImportance'] = importance_df
            
            if not importance_df.empty:
                print("\n  ⭐ Top 5 Features Mais Importantes (Random Forest):")
                # Exibe a importância em formato de tabela Markdown
                print(importance_df.head(5).to_markdown(index=False, numalign="left", stralign="left"))


    return resultados


# ------------------ Função para aplicar em todos os CSVs ------------------

def avaliar_modelos_em_tabelas(output_dir: str, output_dir_analise: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Roda os classificadores em todas as tabelas e compara resultados.
    """

    # Busca por arquivos CSV nas pastas de saída
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
        print(f"\n📊 Avaliando classificadores em: {nome}")

        try:
            # 1. Leitura e criação do alvo binário
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
            df_processado, mediana = criar_alvo_binario_alto_risco(df)
            print(f"  -> Dados de entrada (linhas): {df.shape[0]}")
            print(f"  -> Limiar de Alto Risco (Mediana de TX_EVASAO > 0 e QT_ING >= 10): {mediana:.2f}")

            if df_processado.shape[0] < 50:
                print(f"  ⚠️ Poucos registros ({df_processado.shape[0]}). Pulando...")
                continue
                
            # Verifica a distribuição da classe para evitar problemas (estratificação é essencial)
            if df_processado['ALTO_RISCO'].nunique() < 2:
                print(f"  ⚠️ Apenas uma classe de 'ALTO_RISCO' encontrada. Pulando...")
                continue

            # 2. Treinamento
            resultados = treinar_classificadores_evasao(df_processado)

            # 3. Coleta de Métricas
            for modelo, metricas in resultados.items():
                resultados_gerais.append({
                    'arquivo': nome,
                    'modelo': modelo,
                    'Accuracy': metricas['Accuracy'],
                    'Precision': metricas['Precision'],
                    'Recall': metricas['Recall'],
                    'F1-score': metricas['F1-score'],
                    'ROC-AUC': metricas['ROC-AUC']
                })
                
                if modelo == 'RandomForest' and 'FeatureImportance' in metricas:
                    importances_gerais[nome] = metricas['FeatureImportance']

        except Exception as e:
            print(f"❌ Erro ao processar {nome}: {e}")

    df_resultados = pd.DataFrame(resultados_gerais)

    if not df_resultados.empty:
        print("\n✅ Comparativo final de desempenho dos Classificadores:\n")
        # Ordena por ROC-AUC para melhor visualização do desempenho
        print(df_resultados.sort_values(by=['arquivo', 'ROC-AUC'], ascending=[True, False]).to_markdown(index=False, numalign="left", stralign="left"))

    return df_resultados, importances_gerais


# ------------------ Exemplo de uso ------------------

if __name__ == '__main__':
    df_resultados, importances_finais = avaliar_modelos_em_tabelas(
        OUTPUT_DIR,
        OUTPUT_DIR_ANALISE
    )

    # Salva as métricas comparativas
    if not df_resultados.empty:
        df_resultados.to_csv(
            os.path.join(OUTPUT_DIR, 'comparativo_classificadores_evasao.csv'),
            index=False,
            sep=';',
            encoding='utf-8'
        )
        print("\n📁 Resultados de métricas salvos em: 'microdados_filtrados/comparativo_classificadores_evasao.csv'")
        
    # Salva as importâncias de features do Random Forest
    for filename, df_imp in importances_finais.items():
        df_imp.to_csv(
            os.path.join(OUTPUT_DIR, f'IMPORTANCE_RF_CLASS_{filename}.csv'),
            index=False,
            sep=';',
            encoding='utf-8'
        )
        print(f"📁 Importância de features para {filename} salva.")