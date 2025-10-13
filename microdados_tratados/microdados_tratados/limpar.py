import pandas as pd

# Caminhos dos arquivos
arquivo_entrada = r"microdados_tratados\microdados_tratados\L_T_MICRODADOS_CADASTRO_CURSOS_2021.csv"
arquivo_saida = "2021_DadosInepLetras.csv"

# Ler o arquivo CSV com separador ";" e codificação latin1
df = pd.read_csv(arquivo_entrada, encoding="utf-8", sep=";")

# Filtrar apenas cursos de Letras
filtro_letras = df["NO_CURSO"].str.contains("letras", case=False, na=False)

# Filtrar apenas cursos com TP_GRAU_ACADEMICO == 2.0
filtro_grau = df["TP_GRAU_ACADEMICO"] == 2.0

# Aplicar os filtros
df_filtrado = df[filtro_letras & filtro_grau]

# Colunas que serão removidas
colunas_to_drop = [
    'NU_ANO_CENSO', 'NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO',
    'NO_CURSO', 'NO_CINE_ROTULO', 'NO_CINE_AREA_GERAL',
    'NO_CINE_AREA_ESPECIFICA', 'NO_CINE_AREA_DETALHADA'
]

# Remover as colunas selecionadas (apenas as que existirem no DataFrame)
df_final = df_filtrado.drop(columns=[c for c in colunas_to_drop if c in df_filtrado.columns])
df_final = df_filtrado.dropna()

# Salvar o DataFrame final no arquivo CSV
df_final.to_csv(arquivo_saida, index=False)

print(f"Arquivo limpo gerado com sucesso: {arquivo_saida}")
