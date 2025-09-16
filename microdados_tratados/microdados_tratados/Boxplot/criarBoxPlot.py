import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Carregar os dados
df = pd.read_excel("dados.xlsx")

# Variável numérica alvo
variavel_numerica = "taxa_de_evasao"

# Lista fixa de variáveis categóricas
variaveis_categoricas = [
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_REDE",
    "IN_CAPITAL",
    "CO_REGIAO",
    "CO_UF"
]

# Criar pasta de saída se não existir
output_dir = "imagensBoxPlots"
os.makedirs(output_dir, exist_ok=True)

# Gerar boxplots para cada variável categórica
for col in variaveis_categoricas:
    if col in df.columns:  # garante que a coluna existe no dataset
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y=variavel_numerica, data=df)
        plt.title(f"Boxplot de {variavel_numerica} por {col}")
        plt.xticks(rotation=45)
        plt.ylim(0, 20)  # 🔥 fixa limite do eixo y de 0 até 20
        plt.tight_layout()

        # Caminho do arquivo de saída
        filepath = os.path.join(output_dir, f"boxplot_{col}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()  # fecha a figura para economizar memória
        print(f"✅ Boxplot salvo: {filepath}")
    else:
        print(f"⚠️ A coluna {col} não foi encontrada no dataset.")
