import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Optional, Union, Tuple # Importando tipos para clareza

# --- Configurações de Diretório (Mantidas) ---
INPUT_DIR = 'microdados'
OUTPUT_DIR = 'microdados_filtrados'
OUTPUT_DIR_ANALISE = os.path.join(OUTPUT_DIR, 'tabelas_analise') 

# Cria as pastas de saída se elas não existirem (Mantidas)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANALISE).mkdir(parents=True, exist_ok=True)

# --- Definição de Colunas (Mantidas) ---

colunas_a_remover_especificas: List[str] =[
    'NU_ANO_CENSO', 'SG_UF', 'CO_IES', 'QT_SIT_TRANCADA',
    'QT_SIT_DESVINCULADO', 'QT_SIT_TRANSFERIDO', 'QT_SIT_FALECIDO'
]

colunas_quantitativas_evasao: List[str] = [
    'QT_ING', 'QT_SIT_DESVINCULADO', 'QT_SIT_TRANSFERIDO', 'QT_SIT_FALECIDO'
]

colunas_categoricas_ies: List[str] = [
    'CO_REGIAO', 'CO_UF', 'CO_MUNICIPIO', 'IN_CAPITAL', 
    'TP_DIMENSAO', 'TP_ORGANIZACAO_ACADEMICA', 'TP_REDE', 
    'TP_CATEGORIA_ADMINISTRATIVA', 'IN_COMUNITARIA', 'IN_CONFESSIONAL',
    'CO_CINE_ROTULO', 'CO_CINE_AREA_GERAL', 'CO_CINE_AREA_ESPECIFICA',
    'CO_CINE_AREA_DETALHADA', 'TP_GRAU_ACADEMICO', 'IN_GRATUITO',
    'TP_MODALIDADE_ENSINO', 'TP_NIVEL_ACADEMICO'
]
colunas_analise: List[str] = [col for col in colunas_categoricas_ies + colunas_quantitativas_evasao if col in colunas_categoricas_ies or col in colunas_quantitativas_evasao]


# --- Função de Filtragem e Cálculo (Para Tabela Completa + TX_EVASAO) ---

def filter_and_calculate_evasao(
    df_input: pd.DataFrame, 
    colunas_quant: List[str], 
    cols_drop_especificas: List[str]
) -> Optional[pd.DataFrame]:
    """
    Filtra o DataFrame por Licenciatura em Letras, calcula a TX_EVASAO
    e remove as colunas especificadas.
    """
    
    # 1. Filtro Inicial: Licenciatura (2) e Letras
    df_filtrado: pd.DataFrame = df_input.copy()
    
    if 'TP_GRAU_ACADEMICO' in df_filtrado.columns:
        df_filtrado.loc[:, 'TP_GRAU_ACADEMICO'] = pd.to_numeric(
            df_filtrado['TP_GRAU_ACADEMICO'], errors='coerce'
        )
    
    if 'TP_GRAU_ACADEMICO' in df_filtrado.columns and 'NO_CURSO' in df_filtrado.columns:
        df_filtrado = df_filtrado.loc[
            (df_filtrado['TP_GRAU_ACADEMICO'] == 2) & 
            (df_filtrado['NO_CURSO'].astype(str).str.contains('LETRAS', case=False, na=False))
        ].copy()
    else:
        print("  -> Colunas 'TP_GRAU_ACADEMICO' ou 'NO_CURSO' ausentes para filtragem de Licenciatura em Letras.")
        return None
    
    if df_filtrado.empty:
        print("  -> DataFrame vazio após filtragem de Licenciatura em Letras.")
        return None

    # 2. Pré-processamento das colunas quantitativas para cálculo
    cols_to_process: List[str] = [col for col in colunas_quant if col in df_filtrado.columns]
    
    for col in cols_to_process:
        # Preenche com 0 para o cálculo e para o filtro de volume subsequente
        df_filtrado.loc[:, col] = pd.to_numeric(df_filtrado[col], errors='coerce').fillna(0)
    
    # 3. Cálculo da taxa de evasão
    print("  -> Calculando taxa de evasão (TX_EVASAO)...")
    numerador = df_filtrado['QT_SIT_DESVINCULADO'] + df_filtrado['QT_SIT_TRANSFERIDO']
    denominador = df_filtrado['QT_ING'] - df_filtrado['QT_SIT_FALECIDO']

    # Proteção contra Divisão por Zero
    df_filtrado.loc[:, 'TX_EVASAO'] = np.divide(
        numerador * 100, 
        denominador, 
        out=np.zeros_like(numerador, dtype=float), 
        where=denominador > 0
    )
    # A linha np.divide já garante TX_EVASAO = 0 se denominador <= 0.
    df_filtrado.loc[:, 'TX_EVASAO'] = df_filtrado['TX_EVASAO'].round(2)

    # 4. FILTRO DE VOLUME REVISADO: Denominador da TX_EVASAO deve ser >= 10
    QT_DENOMINADOR_MINIMO = 10 # <--- NOVO LIMITE DE VOLUME MÍNIMO
    rows_before_volume_filter = df_filtrado.shape[0]
    
    # Aplica o filtro de volume (Filtros 1 e 2 combinados)
    df_filtrado = df_filtrado.loc[denominador >= QT_DENOMINADOR_MINIMO].copy()
    
    print(f"  -> {rows_before_volume_filter - df_filtrado.shape[0]} linhas removidas por (QT_ING - QT_SIT_FALECIDO) < {QT_DENOMINADOR_MINIMO}.")
    
    if df_filtrado.empty:
        print("  -> DataFrame vazio após filtro de volume. Pulando salvamento.")
        return None
    
    # 5. Remoção de Colunas (Filtros 4 e 5: Mantido)
    
    # 5a. Drop todas as colunas que começam com NO_
    cols_drop_no: List[str] = [col for col in df_filtrado.columns if col.startswith('NO_')]
    
    # 5b. Drop colunas específicas 
    cols_to_drop_final: List[str] = list(set(cols_drop_no + cols_drop_especificas))
    
    cols_present_to_drop: List[str] = [col for col in cols_to_drop_final if col in df_filtrado.columns]

    df_output: pd.DataFrame = df_filtrado.drop(columns=cols_present_to_drop, errors='ignore')
    
    print(f"  -> Colunas removidas (NO_*, {', '.join(cols_drop_especificas)}): {len(cols_present_to_drop)}")
    
    return df_output

# --- Funções process_analysis_table e main (Mantidas inalteradas) ---

def process_analysis_table(df_base: pd.DataFrame, cols_to_select: List[str], quant_cols_subset: List[str]) -> Optional[pd.DataFrame]:
    """
    Prepara a Tabela de Análise para o modelo, selecionando as colunas
    e garantindo que apenas linhas completas para o modelo sejam mantidas.
    """
    
    print("  -> Preparando Tabela para Análise (Colunas Selecionadas)...")

    cols_in_df: List[str] = [col for col in cols_to_select if col in df_base.columns]
    
    if 'TX_EVASAO' not in df_base.columns:
        print("  -> ERRO: Coluna 'TX_EVASAO' ausente no DataFrame base.")
        return None
    
    df_selecionado: pd.DataFrame = df_base[cols_in_df + ['TX_EVASAO']].copy()
    df_limpo: pd.DataFrame = df_selecionado
    
    if df_limpo.empty:
        print("  -> DataFrame vazio após limpeza. Pulando salvamento de análise.")
        return None
    
    print(f"  -> Tabela de Análise final com {df_limpo.shape[0]} linhas e {df_limpo.shape[1]} colunas.")

    return df_limpo

def main() -> None:
    csv_files: List[str] = glob.glob(os.path.join(INPUT_DIR, '*.CSV'))

    if not csv_files:
        print(f"Nenhum arquivo CSV encontrado na pasta '{INPUT_DIR}'.")
        print(f"Por favor, certifique-se de que seus arquivos estão em: {os.path.abspath(INPUT_DIR)}")
        return

    for file_path in csv_files:
        filename: str = os.path.basename(file_path)
        print(f"\n--- Processando Arquivo: {filename} ---")
        
        try:
            # 1. Lê os dados
            df_original: pd.DataFrame = pd.read_csv(file_path, encoding='latin-1', sep=';', low_memory=False)
            print(f"  -> Arquivo carregado com {df_original.shape[0]} linhas e {df_original.shape[1]} colunas.")
            
            # 1b. dropna() - Remove linhas com qualquer valor vazio (Filtro 1)
            rows_before_drop = df_original.shape[0]
            df_cleaned = df_original.dropna().copy()
            rows_after_drop = df_cleaned.shape[0]
            print(f"  -> {rows_before_drop - rows_after_drop} linhas removidas por NaN (Filtro 1).")
            
            if df_cleaned.empty:
                print("  -> DataFrame vazio após dropna(). Pulando processamento.")
                continue

            # ETAPA A: FILTRAGEM DE LINHAS, CÁLCULO E REMOÇÃO DE COLUNAS (PARA TABELA COMPLETA)

            df_filtrado_completo: Optional[pd.DataFrame] = filter_and_calculate_evasao(
                df_cleaned, 
                colunas_quantitativas_evasao, 
                colunas_a_remover_especificas
            )
            
            if df_filtrado_completo is None or df_filtrado_completo.empty:
                print("  -> Nenhum dado restante após filtragem. Pulando salvamento.")
                continue

            # 2. Salva Tabela Completa Filtrada
            output_full_filename: str = os.path.join(OUTPUT_DIR, f'COMPLETO_LETRAS_LIC_{filename}')
            df_filtrado_completo.to_csv(output_full_filename, index=False, encoding='latin-1', sep=';')
            print(f"  -> Tabela COMPLETA (c/ TX_EVASAO, c/ features) salva em: {output_full_filename} ({df_filtrado_completo.shape[0]} linhas)")

            # ETAPA B: APLICAÇÃO DE FILTROS DE COLUNA E SALVAMENTO DA TABELA DE ANÁLISE (REDUZIDA)
            df_analise: Optional[pd.DataFrame] = process_analysis_table(
                df_filtrado_completo, 
                colunas_analise, 
                colunas_quantitativas_evasao
            )
            
            if df_analise is not None:
                # 3. Salva Tabela de Análise (Colunas Selecionadas + TX_EVASAO)
                output_analysis_filename: str = os.path.join(OUTPUT_DIR_ANALISE, f'ANALISE_EVASAO_{filename}')
                df_analise.to_csv(output_analysis_filename, index=False, encoding='latin-1', sep=';')
                print(f"  -> Tabela de ANÁLISE (Reduzida) salva em: {output_analysis_filename} ({df_analise.shape[0]} linhas)")
                
        except Exception as e:
            print(f"ERRO ao processar o arquivo {filename}: {e}")

    print("\nProcessamento de todos os arquivos concluído.")

main()