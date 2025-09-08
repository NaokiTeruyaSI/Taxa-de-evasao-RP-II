import pandas as pd

# Caminho do arquivo CSV original
arquivo_entrada = "MICRODADOS_CADASTRO_CURSOS_2023.csv"
arquivo_saida = "L_T_MICRODADOS_CADASTRO_CURSOS_2023.csv"

# Ler o arquivo com separador ";" e codificação latin1, usando low_memory=False
df = pd.read_csv(arquivo_entrada, encoding="latin1", sep=";", low_memory=False)

# Filtrar apenas cursos de Letras
filtro_letras = df["NO_CURSO"].str.contains("letras", case=False, na=False)

# Filtrar apenas cursos com TP_GRAU_ACADEMICO == 2.0
filtro_grau = df["TP_GRAU_ACADEMICO"] == 2.0

# Aplicar filtros
df_filtrado = df[filtro_letras & filtro_grau]

# Apagar colunas nas quais todas as linhas têm o mesmo valor
df_filtrado = df_filtrado.loc[:, df_filtrado.nunique() > 1]

# Limpar linhas que possuem valores vazios
df_filtrado = df_filtrado.dropna()

# Salvar o DataFrame filtrado novamente
df_filtrado.to_csv(arquivo_saida, index=False, sep=";", encoding="utf-8")

print(f"Arquivo limpo, colunas removidas e linhas com valores vazios geradas com sucesso: {arquivo_saida}")
