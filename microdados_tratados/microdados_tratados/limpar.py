import pandas as pd

# Caminho do arquivo CSV original
arquivo_entrada = "T_MICRODADOS_CADASTRO_CURSOS_2021.csv"
arquivo_saida = "L_T_MICRODADOS_CADASTRO_CURSOS_2021.csv"

# Ler o arquivo com separador ";" e codificação latin1
df = pd.read_csv(arquivo_entrada, encoding="latin1", sep=";")

# Filtrar apenas cursos de Letras
filtro_letras = df["NO_CURSO"].str.contains("letras", case=False, na=False)

# Filtrar apenas cursos com TP_GRAU_ACADEMICO == 2.0
filtro_grau = df["TP_GRAU_ACADEMICO"] == 2.0

# Aplicar filtros
df_filtrado = df[filtro_letras & filtro_grau]

# Salvar no mesmo encoding (latin1) → Excel lê corretamente
df_filtrado.to_csv(arquivo_saida, index=False, sep=";", encoding="latin1")

colunas_to_drop =['NU_ANO_CENSO', 'NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO', 
                  'NO_CURSO', 'NO_CINE_ROTULO', 'NO_CINE_AREA_GERAL', 'NO_CINE_AREA_ESPECIFICA', 
                  'NO_CINE_AREA_DETALHADA']

print(f"Arquivo limpo gerado com sucesso: {arquivo_saida}")
