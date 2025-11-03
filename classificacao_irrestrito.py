import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# NOVO: Importa SimpleImputer
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- Configurações de Diretório ---
INPUT_DIR = 'microdados'
# Onde os arquivos COMPLETO_LETRAS_LIC_*.CSV estão
OUTPUT_DIR_ORIGINAL = 'microdados_filtrados' 
# Novo diretório para salvar os resultados do teste completo
OUTPUT_DIR_COMPLETO = 'microdados_completo_class' 
OUTPUT_DIR_ANALISE_COMPLETA = os.path.join(OUTPUT_DIR_COMPLETO, 'tabelas_analise') 

# Cria as pastas de saída
Path(OUTPUT_DIR_COMPLETO).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE_COMPLETA).mkdir(parents=True, exist_ok=True)

# Colunas a serem explicitamente excluídas do conjunto de features preditoras
COLUNAS_PARA_EXCLUIR: List[str] = [
    # Identificadores que não devem ser usados como features
    'CO_UF_IES', 'CO_MUNICIPIO_IES', 'CO_IES', 'CO_CINE_AREA', 'CO_OCDE_AREA', 
    'NO_REGIAO_IES', 'NO_UF_IES', 'NO_MUNICIPIO_IES', 'NO_REGIAO', 'NO_UF',
    'NO_MUNICIPIO', 'SG_UF_IES', 'SG_UF', 
    # Colunas que podem vazar a informação ou não são features de curso
    'ANO_BASE', 'DS_ORGANIZACAO_ACADEMICA', 'DS_CATEGORIA_ADMINISTRATIVA',
    'NO_CINE_ROTULO', 'DS_GRAU_ACADEMICO', 'DS_MODALIDADE_ENSINO'
]


def criar_alvo_binario_alto_risco(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Cria a variável alvo binária ALTO_RISCO (1 se TX_EVASAO >= mediana).
    """
    if 'TX_EVASAO' not in df.columns or 'QT_ING' not in df.columns:
        raise ValueError("Colunas 'TX_EVASAO' ou 'QT_ING' ausentes para criar o alvo binário.")
    
    df['TX_EVASAO'] = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
    df['QT_ING'] = pd.to_numeric(df['QT_ING'], errors='coerce').fillna(0) 
    # O dropna essencial foi feito antes, aqui apenas garantimos que TX_EVASAO é numérico e sem NaN (o que já deve ser verdade)
    df_clean = df.dropna(subset=['TX_EVASAO']).copy() 
    
    if df_clean.empty:
        return df_clean, 0.0

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
        
    df_clean['ALTO_RISCO'] = (df_clean['TX_EVASAO'] >= mediana).astype(int)
    
    df_clean = df_clean.drop(columns=['TX_EVASAO'])
    
    return df_clean, mediana


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

    except AttributeError:
        # Apenas ignora se o modelo não tiver feature_importances_ (ex: MLP, LogReg)
        return pd.DataFrame()


def treinar_classificadores_evasao(df: pd.DataFrame) -> Dict[str, Any]:
    """Treina RandomForest, MLP e LogisticRegression para prever ALTO_RISCO."""
    if 'ALTO_RISCO' not in df.columns:
        raise ValueError("Coluna 'ALTO_RISCO' não encontrada no DataFrame.")

    print(f"  -> Iniciando treinamento com {df.shape[0]} registros.")
    y = df['ALTO_RISCO']
    X = df.drop(columns=['ALTO_RISCO']).copy()

    # Identifica colunas (agora X contém todas as features candidatas)
    colunas_numericas = [c for c in X.columns if c.startswith('QT_')]
    colunas_categoricas = [c for c in X.columns if not c.startswith('QT_')]

    # Imputa NaNs nas colunas categóricas antes de converter para string e usar OrdinalEncoder
    for c in colunas_categoricas:
        X[c] = X[c].astype(str).fillna('VALOR_AUSENTE')

    # Pré-processamento OTIMIZADO com Imputer
    
    # 1. Pipeline para Numéricas: Imputer (Mediana) -> Scaler
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Pipeline para Categóricas: OrdinalEncoder
    # Como já imputamos o NaN para string 'VALOR_AUSENTE', o encoder pode processá-lo
    categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, colunas_numericas),
        ('cat', categorical_transformer, colunas_categoricas)
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
        
        if nome == 'RandomForest':
            importance_df = get_feature_importances(pipe, all_features)
            resultados[nome]['FeatureImportance'] = importance_df
            
            if not importance_df.empty:
                print("\n  ⭐ Top 5 Features Mais Importantes (Random Forest):")
                print(importance_df.head(5).to_markdown(index=False, numalign="left", stralign="left"))


    return resultados


# ------------------ FUNÇÃO CENTRAL DE CONCATENAÇÃO E AVALIAÇÃO (COMPLETO) ------------------

def concatenar_e_avaliar_completo(output_dir_original: str, output_dir_completo: str, colunas_excluir: List[str]):
    """
    Concatena todas as tabelas COMPLETO_LETRAS_LIC_*.CSV, usa todas as colunas
    disponíveis (exceto identificadores/vazamento) e treina os classificadores no mega-dataset.
    """
    csv_paths = glob.glob(os.path.join(output_dir_original, 'COMPLETO_LETRAS_LIC_*.CSV'))
    all_data = []
    
    print(f"🔍 Encontrados {len(csv_paths)} arquivos para concatenação com todas as colunas.")

    if not csv_paths:
        print(f"Nenhum CSV COMPLETO encontrado na pasta '{output_dir_original}'.")
        return

    # 1. Concatenação e Filtragem
    for csv_path in csv_paths:
        nome_completo = os.path.basename(csv_path)

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
            
            if 'TX_EVASAO' in df.columns:
                # Identifica TODAS as colunas que podem ser features, mais a TX_EVASAO
                cols_to_keep = [col for col in df.columns if col not in colunas_excluir]
                
                # Certifica-se de que a TX_EVASAO está presente para o cálculo do alvo
                if 'TX_EVASAO' not in cols_to_keep:
                    cols_to_keep.append('TX_EVASAO')
                    
                df_filtrado = df[cols_to_keep].copy()
                all_data.append(df_filtrado)
            else:
                print(f"  ❌ Coluna 'TX_EVASAO' ausente. Pulando {nome_completo}.")
                
        except Exception as e:
            print(f"❌ Erro ao processar o arquivo {nome_completo}: {e}")

    if not all_data:
        print("Nenhum dado válido para processamento.")
        return

    df_mega = pd.concat(all_data, ignore_index=True)

    print(f"\n✅ Concatenação concluída. O dataset possui {df_mega.shape[1]} colunas candidatas e {df_mega.shape[0]} linhas.")

    if df_mega.shape[0] < 50:
        print(f"  ⚠️ Poucos registros ({df_mega.shape[0]}). Não é possível treinar.")
        return

    # CORREÇÃO CRÍTICA: Aplicar dropna APENAS nas colunas essenciais para o ALVO/VOLUME
    COLUNAS_ESSENCIAIS_ALVO = ['TX_EVASAO', 'QT_ING'] 
    
    rows_before_drop_mega = df_mega.shape[0]
    df_mega = df_mega.dropna(subset=COLUNAS_ESSENCIAIS_ALVO).copy()
    rows_after_drop_mega = df_mega.shape[0]
    
    print(f"  -> {rows_before_drop_mega - rows_after_drop_mega} linhas removidas por NaN em '{', '.join(COLUNAS_ESSENCIAIS_ALVO)}'.")


    # 2. Criação do Alvo Binário e Treinamento
    df_processado, mediana = criar_alvo_binario_alto_risco(df_mega)
    
    print(f"  -> Total de linhas após dropna ESSENCIAL e processamento: {df_processado.shape[0]}")
    print(f"  -> Limiar de Alto Risco (Mediana de TX_EVASAO > 0 e QT_ING >= 10): {mediana:.2f}")

    if df_processado.empty:
        print(f"  ⚠️ DataFrame vazio após processamento. Não é possível treinar.")
        return
        
    if df_processado['ALTO_RISCO'].nunique() < 2:
        print(f"  ⚠️ Apenas uma classe de 'ALTO_RISCO' encontrada. Não é possível treinar.")
        return
    
    # 3. Treinamento e Coleta de Métricas
    resultados = treinar_classificadores_evasao(df_processado)

    # 4. Processamento e Salvamento de Resultados
    resultados_gerais = []
    importances_finais = {}
    
    output_filename = 'MEGA_COMPLETO'
    
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
        print("\n✅ Comparativo final de desempenho (MEGA-DATASET COMPLETO):\n")
        print(df_resultados.sort_values(by='ROC-AUC', ascending=False).to_markdown(index=False, numalign="left", stralign="left"))
        
        resultados_path = os.path.join(output_dir_completo, 'comparativo_classificadores_mega_completo.csv')
        df_resultados.to_csv(resultados_path, index=False, sep=';', encoding='utf-8')
        print(f"\n📁 Resultados de métricas MEGA-DATASET COMPLETO salvos em: {resultados_path}")

    for filename, df_imp in importances_finais.items():
        imp_path = os.path.join(output_dir_completo, f'IMPORTANCE_RF_CLASS_{filename}.csv')
        df_imp.to_csv(imp_path, index=False, sep=';', encoding='utf-8')
        print(f"📁 Importância de features MEGA-DATASET COMPLETO para {filename} salva.")


# ------------------ Execução Principal ------------------

if __name__ == '__main__':
    concatenar_e_avaliar_completo(
        OUTPUT_DIR_ORIGINAL,
        OUTPUT_DIR_COMPLETO,
        COLUNAS_PARA_EXCLUIR
    )