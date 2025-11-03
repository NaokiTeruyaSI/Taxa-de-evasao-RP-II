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

# --- Configurações de Diretório Otimizadas ---
INPUT_DIR = 'microdados'
OUTPUT_DIR_ORIGINAL = 'microdados_filtrados' # Diretório onde os arquivos COMPLETO_LETRAS_LIC_*.CSV estão
OUTPUT_DIR_OTIMIZADO = 'microdados_otimizados_class' # Diretório para salvar os resultados
OUTPUT_DIR_ANALISE_OTIMIZADA = os.path.join(OUTPUT_DIR_OTIMIZADO, 'tabelas_analise') 

# Cria as pastas de saída otimizadas
Path(OUTPUT_DIR_OTIMIZADO).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE_OTIMIZADA).mkdir(parents=True, exist_ok=True)

# --- Definição do Conjunto de Features Otimizadas (31 Colunas) ---
# 30 Preditoras + TX_EVASAO
COLUNAS_OTIMIZADAS_CLASSIFICACAO: List[str] = [
    'CO_MUNICIPIO', 'CO_CURSO', 'QT_SIT_TRANCADA', 'QT_MAT', 'QT_MAT_NACBRAS', 
    'CO_UF', 'CO_CINE_ROTULO', 'QT_MAT_PROCESCPUBLICA', 'QT_MAT_FEM', 'QT_MAT_BRANCA',
    'CO_REGIAO', 'QT_ING_PROCESCPUBLICA', 'TP_ORGANIZACAO_ACADEMICA', 'QT_ING_FEM', 
    'QT_ING_VESTIBULAR', 'QT_ING_BRANCA', 'QT_ING_VG_NOVA', 'QT_MAT_MASC', 
    'QT_MAT_PARDA', 'QT_MAT_18_24', 'QT_ING', 'QT_ING_NACBRAS', 'QT_ING_18_24', 
    'QT_ING_PARDA', 'QT_MAT_FINANC', 'QT_ING_MASC', 'QT_MAT_25_29', 'QT_MAT_40_49', 
    'QT_MAT_CORND', 'QT_MAT_FINANC_NREEMB', 
    # Alvo original (para leitura)
    'TX_EVASAO' 
]


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
    QT_ING_MINIMO = 1
    
    evasao_filtrada_para_mediana = df_clean.loc[
        (df_clean['TX_EVASAO'] > 0) & 
        (df_clean['QT_ING'] >= QT_ING_MINIMO),
        'TX_EVASAO'
    ]
    
    if evasao_filtrada_para_mediana.empty:
        mediana = 0.0
        print(f"  -> Aviso: Nenhum curso com TX_EVASAO > 0 e QT_ING >= {QT_ING_MINIMO}. Limiar de Alto Risco será 0.0.")
    else:
        mediana = evasao_filtrada_para_mediana.median()
    
    mediana=mediana

    # 3. Cria o alvo binário
    df_clean['ALTO_RISCO'] = (df_clean['TX_EVASAO'] >= mediana).astype(int)
    
    # Remove a TX_EVASAO original para que ela não vaze como feature
    df_clean = df_clean.drop(columns=['TX_EVASAO'])

    return df_clean, mediana


# --- Funções de Feature Importance e Treinamento (Mantidas) ---

def get_feature_importances(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """Extrai e mapeia a importância das features de um modelo Random Forest Classifier."""
    try:
        model = pipeline.named_steps['model']
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return feature_importance_df

    except AttributeError as e:
        if hasattr(pipeline.named_steps.get('model'), 'feature_importances_'):
            pass 
        else:
            print(f"⚠️ Aviso: Não foi possível extrair feature importance: {e}")
        return pd.DataFrame()


def treinar_classificadores_evasao(df: pd.DataFrame) -> Dict[str, Any]:
    """Treina RandomForest, MLP e LogisticRegression para prever ALTO_RISCO."""
    if 'ALTO_RISCO' not in df.columns:
        raise ValueError("Coluna 'ALTO_RISCO' não encontrada no DataFrame.")

    print(f"  -> Iniciando treinamento com {df.shape[0]} registros.")
    y = df['ALTO_RISCO']
    X = df.drop(columns=['ALTO_RISCO']).copy()

    colunas_numericas = [c for c in X.columns if c.startswith('QT_')]
    colunas_categoricas = [c for c in X.columns if not c.startswith('QT_')]

    for c in colunas_categoricas:
        X[c] = X[c].astype(str)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), colunas_numericas),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), colunas_categoricas)
    ])
    all_features = colunas_numericas + colunas_categoricas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
        
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = y_pred

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
        
        if nome == 'RandomForest':
            importance_df = get_feature_importances(pipe, all_features)
            resultados[nome]['FeatureImportance'] = importance_df
            
            if not importance_df.empty:
                print("\n  ⭐ Top 5 Features Mais Importantes (Random Forest):")
                print(importance_df.head(5).to_markdown(index=False, numalign="left", stralign="left"))


    return resultados


# ------------------ FUNÇÃO CENTRAL DE CONCATENAÇÃO E AVALIAÇÃO ------------------

def concatenar_e_avaliar_otimizado(output_dir_original: str, output_dir_otimizado: str, colunas_otimizadas: List[str]):
    """
    Concatena todas as tabelas COMPLETO_LETRAS_LIC_*.CSV, filtra pelas colunas
    otimizadas e treina os classificadores no mega-dataset.
    """
    csv_paths = glob.glob(os.path.join(output_dir_original, 'COMPLETO_LETRAS_LIC_*.CSV'))
    all_data = []
    
    print(f"🔍 Encontrados {len(csv_paths)} arquivos para concatenação.")

    if not csv_paths:
        print(f"Nenhum CSV COMPLETO encontrado na pasta '{output_dir_original}'.")
        return

    # 1. Concatenação e Filtragem
    for csv_path in csv_paths:
        nome_completo = os.path.basename(csv_path)
        print(f"  -> Lendo e filtrando: {nome_completo}")

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
            
            # Filtra apenas as colunas de interesse
            cols_to_keep = [col for col in colunas_otimizadas if col in df.columns]
            
            if 'TX_EVASAO' in df.columns:
                df_filtrado = df[cols_to_keep].copy()
                # O dropna será feito no mega-dataset para consistência
                all_data.append(df_filtrado)
            else:
                print(f"  ❌ Coluna 'TX_EVASAO' ausente. Pulando {nome_completo}.")
                
        except Exception as e:
            print(f"❌ Erro ao processar o arquivo {nome_completo}: {e}")

    if not all_data:
        print("Nenhum dado válido para processamento.")
        return

    df_mega = pd.concat(all_data, ignore_index=True)
    rows_before_drop = df_mega.shape[0]
    df_mega = df_mega.dropna().copy()
    rows_after_drop = df_mega.shape[0]

    print(f"\n✅ Concatenação concluída. Total de linhas: {rows_after_drop} ({rows_before_drop - rows_after_drop} removidas por NaN).")

    if df_mega.shape[0] < 50:
        print(f"  ⚠️ Poucos registros ({df_mega.shape[0]}). Não é possível treinar.")
        return

    # 2. Criação do Alvo Binário e Treinamento
    df_processado, mediana = criar_alvo_binario_alto_risco(df_mega)
    
    print(f"  -> Limiar de Alto Risco (Mediana de TX_EVASAO > 0 e QT_ING >= 10): {mediana:.2f}")

    if df_processado['ALTO_RISCO'].nunique() < 2:
        print(f"  ⚠️ Apenas uma classe de 'ALTO_RISCO' encontrada. Não é possível treinar.")
        return
    
    # 3. Treinamento e Coleta de Métricas
    resultados = treinar_classificadores_evasao(df_processado)

    # 4. Processamento e Salvamento de Resultados
    resultados_gerais = []
    importances_finais = {}
    
    output_filename = 'MEGA_OTIMIZADO'
    
    for modelo, metricas in resultados.items():
        resultados_gerais.append({
            'arquivo': output_filename,
            'modelo': modelo,
            'Accuracy': metricas['Accuracy'],
            'Precision': metricas['Precision'],
            'Recall': metricas['Recall'],
            'F1-score': metricas['F1-score'],
            'ROC-AUC': metricas['ROC-AUC']
        })
        
        if modelo == 'RandomForest' and 'FeatureImportance' in metricas:
            importances_finais[output_filename] = metricas['FeatureImportance']

    df_resultados = pd.DataFrame(resultados_gerais)

    # 5. Salvamento
    if not df_resultados.empty:
        print("\n✅ Comparativo final de desempenho (MEGA-DATASET OTIMIZADO):\n")
        print(df_resultados.sort_values(by='ROC-AUC', ascending=False).to_markdown(index=False, numalign="left", stralign="left"))
        
        resultados_path = os.path.join(output_dir_otimizado, 'comparativo_classificadores_mega_otimizado.csv')
        df_resultados.to_csv(resultados_path, index=False, sep=';', encoding='utf-8')
        print(f"\n📁 Resultados de métricas MEGA-DATASET salvos em: {resultados_path}")

    for filename, df_imp in importances_finais.items():
        imp_path = os.path.join(output_dir_otimizado, f'IMPORTANCE_RF_CLASS_{filename}.csv')
        df_imp.to_csv(imp_path, index=False, sep=';', encoding='utf-8')
        print(f"📁 Importância de features MEGA-DATASET para {filename} salva.")


# ------------------ Execução Principal ------------------

if __name__ == '__main__':
    concatenar_e_avaliar_otimizado(
        OUTPUT_DIR_ORIGINAL,
        OUTPUT_DIR_OTIMIZADO,
        COLUNAS_OTIMIZADAS_CLASSIFICACAO
    )