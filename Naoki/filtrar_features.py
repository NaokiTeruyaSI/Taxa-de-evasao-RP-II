import pandas as pd
import numpy as np
import os
from typing import Optional 

# ===================================================================
# CONSTANTES GLOBAIS DE CONFIGURAÇÃO
# ===================================================================

DIRETORIO_ENTRADA = "microdados"
DIRETORIO_SAIDA = "microdados_filtrados"
NOME_ARQUIVO_MEGA_DATASET = "mega_dataset_filtrado.csv"

# --- Configurações de Filtro ---
FILTRO_GRAU_ACADEMICO = 2
FILTRO_NOME_CURSO = 'LETRAS'
FILTRO_MIN_INGRESSANTES = 10

# --- COLUNA CRÍTICA ADICIONADA (ESTAVA FALTANDO) ---
# Colunas CRÍTICAS para a análise. Se alguma destas for NaN, a linha é inútil.
COLUNAS_CRITICAS_NAN = [
    'TP_GRAU_ACADEMICO', 'NO_CURSO', 'QT_SIT_DESVINCULADO', 
    'QT_SIT_TRANSFERIDO', 'QT_ING', 'QT_SIT_FALECIDO',
    # Adicione outras colunas que você usa nos cálculos se elas forem essenciais
    'QT_CONC', 'QT_MAT', 'QT_ING_FEM', 'QT_MAT_FEM', 'QT_CONC_FEM',
    'QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39',
    'QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS'
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

def safe_divide(df: pd.DataFrame, numerator_col: str, denominator_col: str) -> pd.Series:
    """
    Calcula a divisão segura (numerador / denominador), preenchento 
    divisões por zero (NaN) com 0.
    """
    if numerator_col in df.columns and denominator_col in df.columns:
        num = pd.to_numeric(df[numerator_col], errors='coerce').fillna(0)
        den = pd.to_numeric(df[denominator_col], errors='coerce')
        
        denominator_safe = den.replace(0, np.nan)
        division = num / denominator_safe
        return division.fillna(0).clip(lower=0.0)
    else:
        print(f"Aviso: Colunas {numerator_col} ou {denominator_col} não encontradas para 'safe_divide'.")
        return pd.Series(0, index=df.index, dtype=float)

# --- NOVA FUNÇÃO AUXILIAR PARA CORRIGIR O ERRO FATAL ---
def get_numeric_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Busca uma coluna de forma segura. Se existir, converte para numérico.
    Se não existir, retorna 0 (escalar, que o pandas aplica a todas as linhas).
    """
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    else:
        # Retorna 0, que será transmitido (broadcasted) para todas as linhas
        # durante operações aritméticas (ex: soma).
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
    # Garante que as colunas críticas existam antes de tentar dropar
    colunas_para_check = [col for col in COLUNAS_CRITICAS_NAN if col in df.columns]
    
    # --- CORREÇÃO DO SettingWithCopyWarning ---
    # Adicionado .copy() para garantir que df_filtrado não seja uma 'view'
    df_filtrado = df.dropna(subset=colunas_para_check).copy()
    
    print(f"Formato depois do dropna (subset): {df_filtrado.shape}")
    
    # 1.2: Garantir que QT_ING seja numérico antes de filtrar
    # Esta operação agora é segura e não gerará o warning
    df_filtrado['QT_ING'] = pd.to_numeric(df_filtrado['QT_ING'], errors='coerce')
    df_filtrado = df_filtrado.dropna(subset=['QT_ING']) 
    
    # 1.3: Aplicar filtros de Grau, Curso e Significância (QT_ING)
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
    
    # --- CORREÇÃO DO ERRO FATAL ---
    # Usando a nova função segura para garantir que as colunas existam
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


def criar_features_engenharia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features (características) a partir das colunas existentes
    para enriquecer análises e melhorar modelos preditivos.
    """
    print("\n--- ETAPA 4: Executando Engenharia de Features ---")

    temp_cols = []

    # ======================================================
    # 1. Indicadores básicos de eficiência e atratividade
    # ======================================================
    df['FEAT_RATIO_CANDIDATO_VAGA'] = safe_divide(df, 'QT_INSCRITO_TOTAL', 'QT_VG_TOTAL')
    df['FEAT_TAXA_OCUPACAO_VAGAS'] = safe_divide(df, 'QT_ING', 'QT_VG_TOTAL')
    # df['FEAT_TAXA_INGRESSO_MATRICULA'] = safe_divide(df, 'QT_ING', 'QT_MAT')

    # ======================================================
    # 2. Perfil de gênero e idade
    # ======================================================
    df['FEAT_PCT_ING_FEM'] = safe_divide(df, 'QT_ING_FEM', 'QT_ING')
    # df['FEAT_PCT_MAT_FEM'] = safe_divide(df, 'QT_MAT_FEM', 'QT_MAT')

    # --- CORREÇÃO DO ERRO FATAL ---
    # Usando a nova função segura para garantir que as colunas existam
    qt_ing_0_17 = get_numeric_col(df, 'QT_ING_0_17')
    qt_ing_18_24 = get_numeric_col(df, 'QT_ING_18_24')
    qt_ing_25_29 = get_numeric_col(df, 'QT_ING_25_29')
    qt_ing_30_34 = get_numeric_col(df, 'QT_ING_30_34')
    qt_ing_35_39 = get_numeric_col(df, 'QT_ING_35_39')
    qt_ing_40_49 = get_numeric_col(df, 'QT_ING_40_49')
    qt_ing_50_59 = get_numeric_col(df, 'QT_ING_50_59')
    qt_ing_60_mais = get_numeric_col(df, 'QT_ING_60_MAIS')

    df['__temp_num_idade'] = (17*qt_ing_0_17 + 21*qt_ing_18_24 + 27*qt_ing_25_29 +
                             32*qt_ing_30_34 + 37*qt_ing_35_39 + 45*qt_ing_40_49 +
                             55*qt_ing_50_59 + 65*qt_ing_60_mais)
    
    df['__temp_den_idade'] = (qt_ing_0_17 + qt_ing_18_24 + qt_ing_25_29 +
                             qt_ing_30_34 + qt_ing_35_39 + qt_ing_40_49 +
                             qt_ing_50_59 + qt_ing_60_mais)
    
    df['FEAT_IDADE_MEDIA_ING'] = safe_divide(df, '__temp_num_idade', '__temp_den_idade')
    temp_cols.extend(['__temp_num_idade', '__temp_den_idade'])

    df['__temp_num_jovem'] = qt_ing_0_17 + qt_ing_18_24
    df['FEAT_PCT_ING_JOVEM'] = safe_divide(df, '__temp_num_jovem', 'QT_ING')
    temp_cols.append('__temp_num_jovem')

    df['__temp_num_idoso'] = qt_ing_50_59 + qt_ing_60_mais
    df['FEAT_PCT_ING_IDOSO'] = safe_divide(df, '__temp_num_idoso', 'QT_ING')
    temp_cols.append('__temp_num_idoso')

    # ======================================================
    # 3. Diversidade Étnico-Racial
    # ======================================================
    # --- CORREÇÃO DO ERRO FATAL ---
    df['__temp_num_pp'] = get_numeric_col(df, 'QT_ING_PRETA') + get_numeric_col(df, 'QT_ING_PARDA')
    df['FEAT_PCT_ING_PRETA_PARDA'] = safe_divide(df, '__temp_num_pp', 'QT_ING')
    temp_cols.append('__temp_num_pp')
    
    df['FEAT_PCT_ING_BRANCA'] = safe_divide(df, 'QT_ING_BRANCA', 'QT_ING')

    df['FEAT_INDICE_DIVERSIDADE'] = 1 - (
        (safe_divide(df, 'QT_ING_BRANCA', 'QT_ING'))**2 +
        (safe_divide(df, 'QT_ING_PRETA', 'QT_ING'))**2 +
        (safe_divide(df, 'QT_ING_PARDA', 'QT_ING'))**2 +
        (safe_divide(df, 'QT_ING_AMARELA', 'QT_ING'))**2 +
        (safe_divide(df, 'QT_ING_INDIGENA', 'QT_ING'))**2
    )

    # ======================================================
    # 4. Inclusão e Ações Afirmativas
    # ======================================================
    df['FEAT_PCT_ING_DEFICIENTE'] = safe_divide(df, 'QT_ING_DEFICIENTE', 'QT_ING')

    # --- CORREÇÃO DO ERRO FATAL ---
    df['__temp_num_reserva'] = (get_numeric_col(df, 'QT_ING_RVREDEPUBLICA') +
                               get_numeric_col(df, 'QT_ING_RVPPI') +
                               get_numeric_col(df, 'QT_ING_RVPOVT') +
                               get_numeric_col(df, 'QT_ING_RVPDEF') +
                               get_numeric_col(df, 'QT_ING_RVSOCIAL_RF'))
    
    df['FEAT_PCT_ING_RESERVA_SOCIAL'] = safe_divide(df, '__temp_num_reserva', 'QT_ING_RESERVA_VAGA')
    temp_cols.append('__temp_num_reserva')
    
    # df['FEAT_PCT_MAT_RESERVA'] = safe_divide(df, 'QT_MAT_RESERVA_VAGA', 'QT_MAT')

    # ======================================================
    # 5. Financiamento e Apoio
    # ======================================================
    df['FEAT_PCT_ING_FINANC'] = safe_divide(df, 'QT_ING_FINANC', 'QT_ING')
    df['FEAT_PCT_ING_COM_FIES'] = safe_divide(df, 'QT_ING_FIES', 'QT_ING_FINANC')
    
    # --- CORREÇÃO DO ERRO FATAL ---
    df['__temp_num_prouni'] = get_numeric_col(df, 'QT_ING_PROUNII') + get_numeric_col(df, 'QT_ING_PROUNIP')
    df['FEAT_PCT_ING_PROUNI'] = safe_divide(df, '__temp_num_prouni', 'QT_ING')
    temp_cols.append('__temp_num_prouni')

    # df['FEAT_PCT_MAT_APOIO_SOCIAL'] = safe_divide(df, 'QT_MAT_APOIO_SOCIAL', 'QT_MAT')
    # df['FEAT_PCT_MAT_FINANC'] = safe_divide(df, 'QT_MAT_FINANC', 'QT_MAT')

    # ======================================================
    # 6. Relações entre etapas (Ingresso → Matrícula → Conclusão)
    # ======================================================
    # df['FEAT_RAZAO_MAT_POR_ING'] = safe_divide(df, 'QT_MAT', 'QT_ING')

    # ======================================================
    # 7. Modalidade e Turno
    # ======================================================
    df['FEAT_PCT_VG_EAD'] = safe_divide(df, 'QT_VG_TOTAL_EAD', 'QT_VG_TOTAL')
    df['FEAT_PCT_VG_NOTURNO'] = safe_divide(df, 'QT_VG_TOTAL_NOTURNO', 'QT_VG_TOTAL')

    # --- CORREÇÃO DO ERRO FATAL ---
    # Usando get_numeric_col para garantir que as colunas existam
    df['__temp_diurno'] = get_numeric_col(df, 'QT_VG_TOTAL_DIURNO')
    df['__temp_noturno'] = get_numeric_col(df, 'QT_VG_TOTAL_NOTURNO')
    df['__temp_ead'] = get_numeric_col(df, 'QT_VG_TOTAL_EAD')
    
    # Criando um dataframe temporário para o idxmax
    df_modalidade = pd.DataFrame({
        'DIURNO': df['__temp_diurno'],
        'NOTURNO': df['__temp_noturno'],
        'EAD': df['__temp_ead']
    })
    
    df['FEAT_DOMINANCIA_MODALIDADE'] = df_modalidade.idxmax(axis=1)
    temp_cols.extend(['__temp_diurno', '__temp_noturno', '__temp_ead'])


    # ======================================================
    # 8. Indicadores Regionais / Institucionais
    # ======================================================
    # --- CORREÇÃO DO ERRO FATAL ---
    df['FEAT_REGIAO_NORTE_NORDESTE'] = get_numeric_col(df, 'CO_REGIAO').isin([1, 2]).astype(int)
    df['FEAT_PUBLICA'] = (get_numeric_col(df, 'TP_REDE') == 1).astype(int)

    # ======================================================
    # Limpeza e retorno
    # ======================================================
    df = df.drop(columns=[col for col in temp_cols if col in df.columns], errors='ignore')
    
    num_feats = len([c for c in df.columns if c.startswith('FEAT_')])
    print(f"{num_feats} novas features (FEAT_*) criadas com sucesso.")
    return df


def remover_colunas_finais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 5: Remove as colunas-fonte da variável alvo (definidas nas
    constantes) para evitar vazamento de dados (data leakage).
    """
    print("\n--- ETAPA 5: Removendo Colunas Finais (Prevenção de Data Leakage) ---")
    
    colunas_para_dropar = [col for col in COLUNAS_DROP_FINAL if col in df.columns]
    df = df.drop(columns=colunas_para_dropar)
    
    print(f"Colunas-alvo removidas: {colunas_para_dropar}")
    return df

# ===================================================================
# EXECUÇÃO PRINCIPAL (PIPELINE)
# ===================================================================

def processar_arquivo(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """
    Executa o pipeline completo de 5 etapas para um único arquivo.
    Retorna o DataFrame processado ou None se falhar.
    """
    try:
        print(f"Lendo arquivo: {caminho_arquivo}")
        # Adicionado dtype='object' para ler tudo como string primeiro
        # Isso evita erros de tipo misto na leitura
        df_bruto = pd.read_csv(caminho_arquivo, sep=";", encoding="latin-1", low_memory=False, dtype='object')
        
        df_filtrado = aplicar_filtros_iniciais(df_bruto)
        
        if df_filtrado.empty:
            print("\nNenhum dado restou após os filtros iniciais. Pulando este arquivo.")
            return None 

        df_sem_colunas = remover_colunas_iniciais(df_filtrado)
        df_com_target = calcular_taxa_evasao(df_sem_colunas)
        df_com_features = criar_features_engenharia(df_com_target)
        df_final = remover_colunas_finais(df_com_features)

        print("\n--- RESULTADO FINAL DO ARQUIVO ---")
        print(f"Formato final do DataFrame: {df_final.shape}")
        
        return df_final 

    except Exception as e:
        print(f"!!! ERRO FATAL ao processar o arquivo {caminho_arquivo}: {e}")
        # import traceback
        # traceback.print_exc() # Descomente para debugar
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
            # Extrai o "ano" do nome do arquivo (ex: microdados_2019.csv → 2019)
            ano_detectado = ''.join(filter(str.isdigit, nome_arquivo)) or "desconhecido"
            df_processado["ANO_BASE"] = ano_detectado

            # Salva versão individual
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
    mega_dataset = pd.concat(lista_dataframes_processados, ignore_index=True)
    caminho_saida_mega = os.path.join(DIRETORIO_SAIDA, NOME_ARQUIVO_MEGA_DATASET)

    mega_dataset.to_csv(caminho_saida_mega, sep=';', index=False, encoding='utf-8')
    print(f"\n✅ Mega-dataset salvo com sucesso em: {caminho_saida_mega}")
    print(f"   ➤ Formato final: {mega_dataset.shape}")
    print("="*80)


if __name__ == "__main__":
    main()