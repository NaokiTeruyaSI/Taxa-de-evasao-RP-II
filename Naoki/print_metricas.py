import os
import pandas as pd
from typing import List
import contextlib

def encontrar_e_imprimir_metricas(start_dir: str, output_file: str = None):
    """
    Percorre recursivamente o 'start_dir', encontra todos os arquivos CSV 
    que começam com 'metricas', e imprime no console ou salva em um arquivo.
    """
    
    # Define o destino da saída
    if output_file:
        f = open(output_file, "w", encoding="utf-8")
        print_target = contextlib.redirect_stdout(f)
    else:
        print_target = contextlib.nullcontext()

    with print_target:
        print("=" * 100)
        print(f"📊 Buscando tabelas de métricas a partir de: {start_dir}")
        print("=" * 100)
        
        arquivos_metricas = []
        
        # 1. Encontrar todos os arquivos de métricas
        for root, dirs, files in os.walk(start_dir):
            dirs[:] = [d for d in dirs if d not in {
                '.git', '__pycache__', 'venv', '.venv', 'microdados'
            }]
            
            for file in files:
                if file.lower().startswith('metricas') and file.lower().endswith('.csv'):
                    arquivos_metricas.append(os.path.join(root, file))

        if not arquivos_metricas:
            print("\nNenhuma tabela de métrica foi encontrada.")
            print("Verifique se os nomes dos arquivos começam com 'metricas_...'.")
            print("=" * 100)
            return

        print(f"\n✅ Encontradas {len(arquivos_metricas)} tabelas de métricas. Imprimindo...\n")

        # 2. Carregar e imprimir cada tabela
        for path in sorted(arquivos_metricas):
            relative_path = os.path.relpath(path, start_dir)
            
            print("-" * 100)
            print(f"📁 Tabela: {relative_path}")
            print("-" * 100)
            
            try:
                df = pd.read_csv(path, sep=';', encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(path, sep=';', encoding='latin-1')
                except Exception as e:
                    print(f"❌ Erro ao ler o arquivo (fallback latin-1): {e}\n")
                    continue
            except Exception as e:
                print(f"❌ Erro ao ler o arquivo: {e}\n")
                continue
                
            if df.empty:
                print("(Tabela vazia)\n")
            else:
                print(df.to_markdown(
                    index=False, 
                    numalign="center", 
                    stralign="center", 
                    floatfmt=".4f"
                ))
                print("\n")
                
            print("-" * 100 + "\n")

        print("=" * 100)
        print("🏁 Busca de métricas concluída.")
        print("=" * 100)

    if output_file:
        f.close()
        print(f"\n✅ Saída salva em: {output_file}")


if __name__ == "__main__":
    diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
    
    # Caminho do arquivo de saída
    arquivo_saida = os.path.join(diretorio_raiz, "metricas_printadas.txt")
    
    encontrar_e_imprimir_metricas(diretorio_raiz, output_file=arquivo_saida)
