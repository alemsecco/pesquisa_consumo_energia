import os
import glob
import pandas as pd
from typing import List


def parse_metadata_and_header(path: str):
	"""Lê o arquivo e retorna (metadata_dict, header_line_index).
	metadata_dict contém Nome, Codigo Estacao, Latitude, Longitude, Altitude se disponíveis."""
	meta = {}
	with open(path, 'r', encoding='utf-8') as f:
		for i, line in enumerate(f):
			line = line.strip()
			# header detection
			if line.startswith('Data Medicao') or line.startswith('Data Medição'):
				return meta, i
			if ':' in line:
				parts = line.split(':', 1)
				key = parts[0].strip()
				val = parts[1].strip()
				meta[key] = val
	return meta, None


def read_temperature_file(path: str) -> pd.DataFrame:
	"""Lê um arquivo de temperatura e retorna um DataFrame com colunas padronizadas."""
	meta, header_row = parse_metadata_and_header(path)
	if header_row is None:
		raise ValueError(f'Cabeçalho não encontrado em {path}')

	# Ler com pandas pulando as linhas de metadata
	df = pd.read_csv(path, sep=';', decimal=',', header=0, skiprows=header_row, encoding='utf-8', engine='python', na_values=['null','NULL','Null'])

	# Padronizar nomes das colunas: remover espaços extras no fim
	df.columns = [c.strip() for c in df.columns]

	# Criar coluna Data no formato YYYYMMDD (sem hífen)
	if 'Data Medicao' in df.columns:
		df['Data'] = pd.to_datetime(df['Data Medicao'], dayfirst=False, errors='coerce').dt.strftime('%Y%m%d')
	else:
		# tentar outras variações
		possible = [c for c in df.columns if 'Data' in c]
		if possible:
			df['Data'] = pd.to_datetime(df[possible[0]], errors='coerce').dt.strftime('%Y%m%d')
		else:
			df['Data'] = None

	# Adicionar metadados extraídos
	df['Cidade'] = meta.get('Nome', '')
	df['CodigoEstacao'] = meta.get('Codigo Estacao', meta.get('CodigoEstacao', ''))
	df['Latitude'] = meta.get('Latitude', '')
	df['Longitude'] = meta.get('Longitude', '')
	df['Altitude'] = meta.get('Altitude', '')
	# Colunas UF e Regiao não estão nos arquivos — preencher com string vazia por enquanto
	df['UF'] = ''
	df['Regiao'] = ''

	# Reordenar colunas: Data, UF, Cidade, Regiao, CodigoEstacao, Latitude, Longitude, Altitude, <medicoes...>
	measurement_cols = [c for c in df.columns if c not in ['Data', 'UF', 'Cidade', 'Regiao', 'CodigoEstacao', 'Latitude', 'Longitude', 'Altitude']]
	cols = ['Data', 'UF', 'Cidade', 'Regiao', 'CodigoEstacao', 'Latitude', 'Longitude', 'Altitude'] + measurement_cols
	return df[cols]


def consolidate_monthly_temperature(input_dir: str, output_path: str) -> pd.DataFrame:
	"""Percorre todos os CSVs em input_dir e concatena em um único CSV salvo em output_path."""
	pattern = os.path.join(input_dir, '*.csv')
	files = sorted(glob.glob(pattern))
	if not files:
		raise FileNotFoundError(f'Nenhum CSV encontrado em {input_dir}')

	dfs: List[pd.DataFrame] = []
	for p in files:
		try:
			df = read_temperature_file(p)
			# adicionar coluna fonte com o nome do arquivo (opcional)
			df['ArquivoFonte'] = os.path.basename(p)
			dfs.append(df)
		except Exception as e:
			print(f'Erro ao processar {p}: {e}')

	if not dfs:
		raise RuntimeError('Nenhum DataFrame válido foi lido')

	# Concatener alinhando colunas (pandas preenche NaNs onde faltar)
	df_all = pd.concat(dfs, ignore_index=True, sort=False)

	# Garantir ordem das colunas: juntar todas as colunas de medição únicas
	front = ['Data', 'UF', 'Cidade', 'Regiao', 'CodigoEstacao', 'Latitude', 'Longitude', 'Altitude', 'ArquivoFonte']
	other_cols = [c for c in df_all.columns if c not in front]
	df_all = df_all[front + other_cols]

	# Salvar CSV com ; e vírgula decimal
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
	df_all.to_csv(output_path, index=False, sep=';', decimal=',')
	return df_all


if __name__ == '__main__':
	# defaults
	base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temperatura', 'mensal'))
	input_dir = base
	output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'temperatura', 'temperatura_mensal_unificado.csv'))

	print(f'Procurando CSVs em: {input_dir}')
	df_all = consolidate_monthly_temperature(input_dir, output)
	print(f'Registros consolidados: {len(df_all)}')
	print(f'Arquivo salvo em: {output}')
