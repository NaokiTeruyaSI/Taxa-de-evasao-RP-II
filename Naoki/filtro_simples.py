import pandas as pd
import numpy as np
import os
from typing import Optional 
import traceback # Para debugar

# ===================================================================
# CONSTANTES GLOBAIS DE CONFIGURAÇÃO
# ===================================================================

DIRETORIO_ENTRADA = "microdados"
DIRETORIO_SAIDA = "microdados_filtrados_simples"
NOME_ARQUIVO_MEGA_DATASET = "mega_dataset_filtrado_simples.csv"

# --- Configurações de Filtro ---
FILTRO_GRAU_ACADEMICO = 2
FILTRO_NOME_CURSO = 'LETRAS'
FILTRO_MIN_INGRESSANTES = 10

# --- COLUNAS CRÍTICAS SIMPLIFICADAS ---
# Apenas o que é essencial para os filtros e o cálculo do alvo
COLUNAS_CRITICAS_NAN = [
    'TP_GRAU_ACADEMICO', 'NO_CURSO', 
    'QT_ING', 'QT_SIT_DESVINCULADO', 
    'QT_SIT_TRANSFERIDO', 'QT_SIT_FALECIDO'
]

# --- Configurações de Remoção de Colunas ---
COLUNAS_DROP_INICIAL = ['NU_ANO_CENSO', 'SG_UF', 'CO_IES']
COLUNAS_DROP_FINAL = [
    'QT_SIT_TRANCADA', 'QT_SIT_DESVINCULADO', 
    'QT_SIT_TRANSFERIDO', 'QT_SIT_FALECIDO'
]

# ===================================================================
# FUNÇÕES AUXILIARES
# ===================================================================

# A função 'safe_divide' foi removida pois não é mais usada.

def get_numeric_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Busca uma coluna de forma segura. Se existir, converte para numérico.
    Se não existir, retorna 0 (escalar).
    """
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    else:
        return 0

# ===================================================================
# ETAPAS DO PIPELINE DE PRÉ-PROCESSAMENTO
# ===================================================================

def aplicar_filtros_iniciais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 1: Aplica os filtros de limpeza (NaN), de negócio (Grau, Curso)
             e de significância (QT_ING).
    """
    print("\n--- ETAPA 1: Aplicando Filtros Iniciais ---")
    
    print(f"Formato antes do dropna: {df.shape}")
    colunas_para_check = [col for col in COLUNAS_CRITICAS_NAN if col in df.columns]
    
    # Adicionado .copy() para garantir que df_filtrado não seja uma 'view'
    df_filtrado = df.dropna(subset=colunas_para_check).copy()
    
    print(f"Formato depois do dropna (subset): {df_filtrado.shape}")
    
    df_filtrado['QT_ING'] = pd.to_numeric(df_filtrado['QT_ING'], errors='coerce')
    df_filtrado = df_filtrado.dropna(subset=['QT_ING']) 
    
    df_filtrado['TP_GRAU_ACADEMICO'] = pd.to_numeric(df_filtrado['TP_GRAU_ACADEMICO'], errors='coerce')
    
    filtro_grau = (df_filtrado['TP_GRAU_ACADEMICO'] == FILTRO_GRAU_ACADEMICO)
    filtro_curso = (df_filtrado['NO_CURSO'].astype(str).str.contains(FILTRO_NOME_CURSO, case=False, na=False))
    filtro_significancia_ing = (df_filtrado['QT_ING'] > FILTRO_MIN_INGRESSANTES)
    
    df_processado = df_filtrado[
        filtro_grau & 
        filtro_curso & 
        filtro_significancia_ing
    ].copy()
    
    print(f"Formato após filtros de Grau, Curso e QT_ING > {FILTRO_MIN_INGRESSANTES}: {df_processado.shape}")
    
    return df_processado


def remover_colunas_iniciais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 2: Remove colunas de nomes (NO_) e outras colunas 
    específicas (definidas nas constantes).
    """
    print("\n--- ETAPA 2: Removendo Colunas Iniciais ---")
    
    colunas_no = [col for col in df.columns if col.startswith('NO_')]
    df = df.drop(columns=colunas_no, errors='ignore')
    print(f"Colunas 'NO_' removidas: {colunas_no}")

    colunas_para_dropar = [col for col in COLUNAS_DROP_INICIAL if col in df.columns]
    df = df.drop(columns=colunas_para_dropar)
    print(f"Colunas específicas removidas: {colunas_para_dropar}")
    
    return df


def calcular_taxa_evasao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 3: Calcula a variável alvo (TX_EVASAO) conforme a fórmula.
    """
    print("\n--- ETAPA 3: Calculando Taxa de Evasão (Target) ---")
    
    qt_desvinculado = get_numeric_col(df, 'QT_SIT_DESVINCULADO')
    qt_transferido = get_numeric_col(df, 'QT_SIT_TRANSFERIDO')
    qt_ing = get_numeric_col(df, 'QT_ING')
    qt_falecido = get_numeric_col(df, 'QT_SIT_FALECIDO')

    numerador = qt_desvinculado + qt_transferido
    denominador = qt_ing - qt_falecido

    df['TX_EVASAO'] = numerador / denominador.replace(0, np.nan)
    df['TX_EVASAO'] = df['TX_EVASAO'].fillna(0)
    df['TX_EVASAO'] = df['TX_EVASAO'].clip(lower=0.0)
    
    print("Coluna 'TX_EVASAO' criada.")
    return df

# ======================================================
# --- ETAPA 4: FUNÇÃO 'criar_features_engenharia' REMOVIDA ---
# ======================================================


def remover_colunas_finais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 5: Remove as colunas-fonte da variável alvo (definidas nas
    constantes) para evitar vazamento de dados (data leakage).
    """
    print("\n--- ETAPA 4: Removendo Colunas Finais (Prevenção de Data Leakage) ---")
    
    colunas_para_dropar = [col for col in COLUNAS_DROP_FINAL if col in df.columns]
    df = df.drop(columns=colunas_para_dropar)
    
    print(f"Colunas-alvo removidas: {colunas_para_dropar}")
    return df

# ===================================================================
# EXECUÇÃO PRINCIPAL (PIPELINE)
# ===================================================================

def processar_arquivo(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """
    Executa o pipeline simplificado de 4 etapas para um único arquivo.
    Retorna o DataFrame processado ou None se falhar.
    """
    try:
        print(f"Lendo arquivo: {caminho_arquivo}")
        df_bruto = pd.read_csv(caminho_arquivo, sep=";", encoding="latin-1", low_memory=False, dtype='object')
        
        df_filtrado = aplicar_filtros_iniciais(df_bruto)
        
        if df_filtrado.empty:
            print("\nNenhum dado restou após os filtros iniciais. Pulando este arquivo.")
            return None 

        df_sem_colunas = remover_colunas_iniciais(df_filtrado)
        df_com_target = calcular_taxa_evasao(df_sem_colunas)
        
        # --- CHAMADA DA ETAPA 4 REMOVIDA ---
        # df_com_features = criar_features_engenharia(df_com_target) 
        
        # A Etapa 5 agora usa 'df_com_target'
        df_final = remover_colunas_finais(df_com_target) 

        print("\n--- RESULTADO FINAL DO ARQUIVO ---")
        print(f"Formato final do DataFrame: {df_final.shape}")
        
        return df_final 

    except Exception as e:
        print(f"!!! ERRO FATAL ao processar o arquivo {caminho_arquivo}: {e}")
        traceback.print_exc() 
        print("Pulando para o próximo arquivo...")
        return None

def main():
    """
    Orquestra a execução do pipeline para cada arquivo CSV no diretório de entrada,
    salvando as versões filtradas individuais e um mega-dataset concatenado ao final.
    """
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

    try:
        nomes_arquivos = sorted([f for f in os.listdir(DIRETORIO_ENTRADA) if f.lower().endswith('.csv')])
    except FileNotFoundError:
        print(f"❌ ERRO: Diretório de entrada não encontrado: {DIRETORIO_ENTRADA}")
        return

    if not nomes_arquivos:
        print(f"⚠️ Nenhum arquivo CSV encontrado em '{DIRETORIO_ENTRADA}'.")
        return

    lista_dataframes_processados = []

    print("="*80)
    print(f"📊 Iniciando processamento de {len(nomes_arquivos)} arquivos CSV...")
    print("="*80)

    for nome_arquivo in nomes_arquivos:
        print(f"\n🔹 PROCESSANDO ARQUIVO: {nome_arquivo}")
        caminho_arquivo = os.path.join(DIRETORIO_ENTRADA, nome_arquivo)

        df_processado = processar_arquivo(caminho_arquivo)

        if df_processado is not None and not df_processado.empty:
            ano_detectado = ''.join(filter(str.isdigit, nome_arquivo)) or "desconhecido"
            df_processado["ANO_BASE"] = ano_detectado

            nome_saida_individual = f"microdados_filtrado_{ano_detectado}.csv"
            caminho_saida_individual = os.path.join(DIRETORIO_SAIDA, nome_saida_individual)
            df_processado.to_csv(caminho_saida_individual, sep=';', index=False, encoding='utf-8')

            print(f"✅ Arquivo filtrado salvo: {caminho_saida_individual}")
            lista_dataframes_processados.append(df_processado)
        else:
            print(f"⚠️ Nenhum dado aproveitável em {nome_arquivo}. Pulando.")

    print("\n========================================================")
    print("🏁 Etapa de processamento individual concluída.")
    print("========================================================")

    if not lista_dataframes_processados:
        print("❌ Nenhum dataset foi gerado. Encerrando sem criar mega-dataset.")
        return

    # Concatenação final
    print(f"\n📦 Concatenando {len(lista_dataframes_processados)} arquivos processados em um mega-dataset...")
    # sort=False previne avisos de desalinhamento de colunas
    mega_dataset = pd.concat(lista_dataframes_processados, ignore_index=True, sort=False)
    caminho_saida_mega = os.path.join(DIRETORIO_SAIDA, NOME_ARQUIVO_MEGA_DATASET)

    mega_dataset.to_csv(caminho_saida_mega, sep=';', index=False, encoding='utf-8')
    print(f"\n✅ Mega-dataset salvo com sucesso em: {caminho_saida_mega}")
    print(f"   ➤ Formato final: {mega_dataset.shape}")
    print("="*80)


if __name__ == "__main__":
    main()