import pandas as pd
import numpy as np
import os

def filter_residencial(in_path: str, out_path: str) -> pd.DataFrame:
	"""Lê CSV com separador ';' e vírgula decimal, filtra linhas onde Classe == 'Residencial'
	e salva o resultado em out_path. Retorna o DataFrame filtrado."""
	# Ler CSV: separador ';' e usar decimal=',' para números com vírgula
	df = pd.read_csv(in_path, sep=';', decimal=',')

	# Normalizar coluna Classe (remover espaços e comparar case-insensitive)
	if 'Classe' not in df.columns:
		raise KeyError("Coluna 'Classe' não encontrada no CSV")

	df['Classe_norm'] = df['Classe'].astype(str).str.strip().str.lower()
	filtro = df['Classe_norm'] == 'residencial'
	df_res = df.loc[filtro].copy()

	# Remover coluna auxiliar
	df_res.drop(columns=['Classe_norm'], inplace=True)

	# Garantir diretório de saída
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	df_res.to_csv(out_path, index=False, sep=';', decimal=',')

	return df_res


if __name__ == '__main__':
	in_file = os.path.join(os.path.dirname(__file__) or '.', 'Dados_abertos_Consumo_Mensal.csv')
	out_file = os.path.join(os.path.dirname(__file__) or '.', 'Dados_residencial.csv')

	print(f"Lendo: {in_file}")
	df_out = filter_residencial(in_file, out_file)
	print(f"Registros totais: {len(df_out)}")
	print(f"Arquivo salvo em: {out_file}")

