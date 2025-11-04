import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# LabelEncoder não é mais necessário para o alvo binário numérico
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import traceback
import time

# --- Configurações de Diretório ---
INPUT_DIR = 'microdados_filtrados_simples'
OUTPUT_DIR_ANALISE = 'analise_classificacao_binaria_simples' # <-- Pasta de saída atualizada
os.makedirs(OUTPUT_DIR_ANALISE, exist_ok=True)


# ------------------ FUNÇÃO DE ALVO (MODIFICADA PARA BINÁRIA) ------------------

def criar_alvo_binario_risco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável alvo binária 'NIVEL_RISCO' (0=Baixo, 1=Alto)
    baseada na mediana (Q2).
    """
    print("\n🔹 [1/3] Criando variável alvo BINÁRIA (Alto/Baixo) baseada em TX_EVASAO...")
    if 'TX_EVASAO' not in df.columns or 'QT_ING' not in df.columns:
        raise ValueError("❌ Colunas 'TX_EVASAO' ou 'QT_ING' ausentes para criar o alvo.")
    
    df['TX_EVASAO'] = pd.to_numeric(df['TX_EVASAO'], errors='coerce')
    df['QT_ING'] = pd.to_numeric(df['QT_ING'], errors='coerce').fillna(0)
    df_clean = df.dropna(subset=['TX_EVASAO']).copy()
    
    if df_clean.empty:
        print("⚠️ Nenhum dado válido após limpeza — retornando vazio.")
        return df_clean

    QT_ING_MINIMO = 10 
    evasao_filtrada_para_mediana = df_clean.loc[
        (df_clean['TX_EVASAO'] > 0) & 
        (df_clean['QT_ING'] >= QT_ING_MINIMO),
        'TX_EVASAO'
    ]
    
    if evasao_filtrada_para_mediana.empty:
        mediana = 0.0
        print(f"⚠️ Nenhum curso com TX_EVASAO > 0 e QT_ING >= {QT_ING_MINIMO}.")
    else:
        # Calcula apenas a mediana (Q2)
        mediana = evasao_filtrada_para_mediana.median()

    # --- MUDANÇA: Usar np.where para criar alvo binário 0 ou 1 ---
    # 1 = Alto Risco (acima da mediana)
    # 0 = Baixo Risco (igual ou abaixo da mediana)
    df_clean['NIVEL_RISCO'] = np.where(df_clean['TX_EVASAO'] > mediana, 1, 0)
    
    df_clean = df_clean.drop(columns=['TX_EVASAO'])
    
    print(f"   ➤ Mediana (Q2) calculada: {mediana:.2f}")
    print(f"   ➤ Classes criadas: {df_clean['NIVEL_RISCO'].value_counts().to_dict()}")
    print("   ➤ (0 = Baixo Risco, 1 = Alto Risco)")
            
    return df_clean


# ------------------ FEATURE IMPORTANCE (MODIFICADA) ------------------

def get_feature_importances(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extrai e mapeia a importância das features de um modelo.
    """
    try:
        model = pipeline.named_steps['model']
        if not hasattr(model, "feature_importances_"):
            if hasattr(model, "coef_"):
                # --- MUDANÇA: Lógica para LogisticRegression binária ---
                # model.coef_ agora é (1, n_features), pegamos o [0]
                importances = np.abs(model.coef_[0])
            else:
                print(f"⚠️ Modelo '{type(model).__name__}' não fornece importâncias de features.")
                return pd.DataFrame()
        else:
            importances = model.feature_importances_
            
        df_importance = (
            pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            .sort_values(by='Importance', ascending=False)
            .reset_index(drop=True)
        )

        print("   ✅ Importâncias de features extraídas com sucesso.")
        return df_importance

    except Exception as e:
        print(f"⚠️ Erro ao extrair feature importance: {e}")
        return pd.DataFrame()


# ------------------ TREINAMENTO DE CLASSIFICADORES (MODIFICADO) ------------------

def treinar_classificadores_evasao(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n🔹 [2/3] Iniciando treinamento dos modelos de classificação...\n")
    start_time = time.time()

    TARGET_COLUMN = 'NIVEL_RISCO'
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"❌ Coluna '{TARGET_COLUMN}' não encontrada no DataFrame.")

    # --- MUDANÇA: LabelEncoder não é mais necessário ---
    y = df[TARGET_COLUMN] # 'y' agora é 0 ou 1
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    # --------------------------------------------------

    # --- Classificação de colunas ---
    print("⚙️ Classificando variáveis (numéricas x categóricas)...")
    colunas_numericas = []
    colunas_categoricas = []
    feat_categoricas_conhecidas = ['FEAT_DOMINANCIA_MODALIDADE']
    ignorar = ['CO_MUNICIPIO', 'CO_CURSO', 'CO_IES']

    for c in X.columns:
        if c in ignorar:
            continue
        if c.startswith(('QT_', 'FEAT_')) and c not in feat_categoricas_conhecidas:
            colunas_numericas.append(c)
        else:
            colunas_categoricas.append(c)

    print(f"   ➤ Numéricas: {len(colunas_numericas)} | Categóricas: {len(colunas_categoricas)}")

    for c in colunas_categoricas:
        X[c] = X[c].astype(str)

    # --- Pipelines ---
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, colunas_numericas),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), colunas_categoricas)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- MUDANÇA: y_test_strings não é mais necessário ---
    # y_test já é numérico (0, 1) e será usado para todas as métricas

    modelos = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced',
            max_depth=15, min_samples_leaf=5),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='tanh', solver='adam',
            max_iter=5000, random_state=42, alpha=0.001),
        'LogisticRegression': LogisticRegression(
            random_state=42, solver='lbfgs', class_weight='balanced',
            max_iter=5000, C=0.1),
        'XGBoost': XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1,
            # --- MUDANÇA: 'scale_pos_weight' para classes desbalanceadas (opcional mas bom) ---
            # Se a classe 1 (Alto Risco) for rara, isso ajuda
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)
    }

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"\n🚀 Treinando modelo: {nome}...")
        pipe = Pipeline([('preprocess', preprocessor), ('model', modelo)])
        t0 = time.time()

        pipe.fit(X_train, y_train) 
        duracao = time.time() - t0
        print(f"   ✅ Modelo '{nome}' treinado em {duracao:.2f}s.")

        y_pred = pipe.predict(X_test)
        
        # --- MUDANÇA: Métricas atualizadas para binário ---
        # Usamos average='binary' e pos_label=1 (Alto Risco)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)

        try:
            # Pega a probabilidade da classe positiva (1 = Alto Risco)
            y_proba = pipe.predict_proba(X_test)[:, 1] 
            auc = roc_auc_score(y_test, y_proba) # Cálculo de AUC binário
        except Exception as e:
            print(f"   ⚠️ Não foi possível calcular ROC-AUC para {nome}: {e}")
            auc = np.nan

        resultados[nome] = {
            'Accuracy': acc,
            'Precision (Alto Risco)': precision, # Label da métrica atualizada
            'Recall (Alto Risco)': recall,       # Label da métrica atualizada
            'F1-score (Alto Risco)': f1,       # Label da métrica atualizada
            'ROC-AUC': auc,                      # Label da métrica atualizada
            'modelo': pipe
        }
        
        if hasattr(modelo, "feature_importances_") or hasattr(modelo, "coef_"):
            imp = get_feature_importances(pipe, colunas_numericas + colunas_categoricas)
            if imp is not None and not imp.empty:
                resultados[nome]['FeatureImportance'] = imp
                print(f"\n⭐ Top 5 Features mais relevantes ({nome}):")
                print(imp.head(5).to_markdown(index=False))

    print(f"\n🧩 Treinamento concluído em {time.time() - start_time:.2f}s total.\n")
    return resultados

# ------------------ MAIN (MODIFICADO) ------------------

def main():
    print("=" * 60)
    print("📊 INÍCIO DA ANÁLISE DE CLASSIFICAÇÃO (BINÁRIA)") # <-- Título atualizado
    print("=" * 60)

    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not all_files:
        print(f"❌ Nenhum arquivo CSV encontrado em '{INPUT_DIR}'.")
        print("   ➤ Certifique-se de que o diretório contém os arquivos de microdados filtrados.")
        return

    print(f"\n📂 Encontrados {len(all_files)} arquivos CSV em '{INPUT_DIR}':")
    for f in all_files:
        print(f"   - {os.path.basename(f)}")

    resultados_gerais = []
    importancias_gerais = []

    for f in all_files:
        print(f"\n{'-'*60}")
        print(f"📄 Processando arquivo: {os.path.basename(f)}")

        try:
            df = pd.read_csv(f, sep=';', encoding='latin-1', low_memory=False)
            print(f"   ➤ Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
        except Exception as e:
            print(f"⚠️ Erro ao ler '{f}': {e}")
            continue

        if df.empty:
            print("⚠️ Arquivo vazio — pulando para o próximo.")
            continue
            
        # # --- AVISO DE DATA LEAKAGE (Mantido) ---
        # features_com_leakage = [col for col in df.columns if '_MAT' in col or '_CONC' in col]
        # if features_com_leakage:
        #     print("\n" + "!"*60)
        #     print("⚠️ AVISO DE DATA LEAKAGE (VAZAMENTO DE DADOS) ⚠️")
        #     print("  Este arquivo contém features baseadas em 'QT_MAT' e 'QT_CONC'.")
        #     print("  Para um modelo preditivo real, elas devem ser removidas do 'filtrar.py'.")
        #     print("!"*60)
        # # --- FIM DO AVISO ---

        try:
            # --- MUDANÇA: Chamando a nova função binária ---
            # Note que 'quartis' não é mais retornado
            df_proc = criar_alvo_binario_risco(df)
            
            if df_proc.shape[0] < 50 or df_proc['NIVEL_RISCO'].nunique() < 2:
                print("⚠️ Dataset insuficiente ou com poucas classes (0 e 1). Pulando arquivo.")
                continue

            # 2️⃣ Treinar classificadores
            resultados = treinar_classificadores_evasao(df_proc)

            # 3️⃣ Coletar métricas
            for nome_modelo, m in resultados.items():
                metrica = {k: v for k, v in m.items() if k not in ['modelo', 'FeatureImportance']}
                metrica['Arquivo'] = os.path.basename(f)
                metrica['Modelo'] = nome_modelo
                resultados_gerais.append(metrica)

                # 4️⃣ Coletar importâncias
                if 'FeatureImportance' in m and not m['FeatureImportance'].empty:
                    imp = m['FeatureImportance'].copy()
                    imp['Arquivo'] = os.path.basename(f)
                    importancias_gerais.append(imp)

            print(f"✅ Arquivo '{os.path.basename(f)}' processado com sucesso.")

        except Exception as e:
            print(f"❌ Erro durante o processamento do arquivo '{os.path.basename(f)}': {e}")
            traceback.print_exc()
            continue

    # ------------------ CONSOLIDAÇÃO FINAL ------------------

    if not resultados_gerais:
        print("\n" + "="*60)
        print("⚠️ Nenhum resultado foi gerado — verifique os arquivos de entrada.")
        print("="*60)
        return

    df_result_final = pd.DataFrame(resultados_gerais)
    
    # --- MUDANÇA: Colunas de métricas binárias atualizadas ---
    col_order = ['Arquivo', 'Modelo', 'Accuracy', 'F1-score (Alto Risco)', 'ROC-AUC', 'Precision (Alto Risco)', 'Recall (Alto Risco)']
    df_result_final = df_result_final.reindex(columns=col_order).sort_values(
        by=['Arquivo', 'ROC-AUC'], ascending=[True, False]
    )

    # --- MUDANÇA: Nomes de arquivos de saída atualizados ---
    output_metrics = os.path.join(OUTPUT_DIR_ANALISE, 'metricas_classificacao_BINARIO_CONSOLIDADO.csv')
    df_result_final.to_csv(output_metrics, sep=';', index=False, encoding='utf-8')

    print("\n📊 RESULTADOS CONSOLIDADOS (ANÁLISE BINÁRIA):\n") # <-- Título atualizado
    print(df_result_final[['Arquivo', 'Modelo', 'Accuracy', 'F1-score (Alto Risco)', 'ROC-AUC']]
          .to_markdown(index=False))

    if importancias_gerais:
        df_import_final = pd.concat(importancias_gerais, ignore_index=True)
        output_importance = os.path.join(OUTPUT_DIR_ANALISE, 'IMPORTANCE_FEATURES_BINARIO_CONSOLIDADO.csv')
        df_import_final.to_csv(output_importance, sep=';', index=False)
        print(f"\n📁 Importância de Features consolidada salva em: {output_importance}")

    print(f"\n📁 Métricas consolidadas salvas em: {output_metrics}")
    print("\n✅ Análise de Classificação Binária Concluída com sucesso!")
    print("=" * 60)


if __name__ == '__main__':
    main()