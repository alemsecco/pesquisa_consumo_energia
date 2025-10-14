import os
import pandas as pd


def monthyear_from_excel_date(s: pd.Series) -> pd.Series:
	# DataExcel in format dd/mm/YYYY
	return pd.to_datetime(s, dayfirst=True, errors='coerce').dt.to_period('M').astype(str)


def monthyear_from_numeric_date(s: pd.Series) -> pd.Series:
	# Data in format YYYYMMDD or numeric like 20250101 or floats like 20040131.0
	# normalize: coerce to numeric, cast to int, then to string YYYYMMDD
	num = pd.to_numeric(s, errors='coerce')
	# dropna handled by to_datetime errors='coerce'
	as_int = num.dropna().astype(int).astype(str)
	res = pd.Series(index=s.index, dtype=object)
	if not as_int.empty:
		parsed = pd.to_datetime(as_int, format='%Y%m%d', errors='coerce').dt.to_period('M').astype(str)
		# map back to original index positions where notna
		res.loc[as_int.index] = parsed.values
	return res


def aggregate_consumption(in_path: str) -> pd.DataFrame:
	# leitura: campo Consumo usa '.' como separador de milhares e ',' como decimal
	df = pd.read_csv(in_path, sep=';', decimal=',', encoding='utf-8')
	# criar coluna MesAno
	if 'DataExcel' in df.columns:
		df['MesAno'] = monthyear_from_excel_date(df['DataExcel'])
	else:
		df['MesAno'] = monthyear_from_numeric_date(df['Data'])

	# garantir colunas
	if 'Regiao' not in df.columns:
		raise KeyError('Coluna Regiao não encontrada em consumo')

	# Consumo pode ter separador decimal vírgula -- já lido com decimal=','
	# Agregar: somar Consumo por MesAno e Regiao
	# limpar separador de milhares e converter decimal
	if 'Consumo' in df.columns:
		# garantir string
		tmp = df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
		df['Consumo'] = pd.to_numeric(tmp, errors='coerce')
	agg = df.groupby(['MesAno', 'Regiao'], dropna=False)['Consumo'].sum().reset_index()
	return agg


def aggregate_temperature(in_path: str) -> pd.DataFrame:
	df = pd.read_csv(in_path, sep=';', decimal=',', encoding='utf-8')

	# Robust MesAno derivation: try common columns and formats
	mesano = None
	if 'Data Medicao' in df.columns:
		# format likely YYYY-MM-DD
		try:
			mesano = pd.to_datetime(df['Data Medicao'], errors='coerce').dt.to_period('M').astype(str)
		except Exception:
			mesano = None
	if mesano is None and 'Data' in df.columns:
		# handle numeric-like values (20040131.0)
		num = pd.to_numeric(df['Data'], errors='coerce')
		idx_valid = num.dropna().index
		mesano = pd.Series(index=df.index, dtype=object)
		if not idx_valid.empty:
			as_int = num.loc[idx_valid].astype(int).astype(str)
			parsed = pd.to_datetime(as_int, format='%Y%m%d', errors='coerce').dt.to_period('M').astype(str)
			mesano.loc[idx_valid] = parsed.values
	if mesano is None:
		possible = [c for c in df.columns if 'Data' in c]
		if possible:
			mesano = pd.to_datetime(df[possible[0]], dayfirst=True, errors='coerce').dt.to_period('M').astype(str)
	if mesano is None:
		raise KeyError('Coluna de data não encontrada em temperatura')
	df['MesAno'] = mesano

	# usar coluna Regiao ou Regiao (você tem 'Regiao' em temperatura_mensal_com_regiao_final.csv)
	if 'Regiao' not in df.columns and 'Regiao' not in df.columns:
		raise KeyError('Coluna Regiao não encontrada em temperatura')

	# selecionar as colunas de interesse para agregação
	meas_cols = [
		'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
		'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
		'PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)',
		'TEMPERATURA MEDIA, MENSAL (AUT)(°C)',
		'VENTO, VELOCIDADE MAXIMA MENSAL (AUT)(m/s)',
		'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)'
	]

	# nem todos arquivos terão todas as colunas; pegar as que existem
	meas_cols = [c for c in meas_cols if c in df.columns]

	# converter medias/numeros
	for c in meas_cols:
		df[c] = pd.to_numeric(df[c], errors='coerce')

	# Agregar: para precipitação total somar, para dias com precip somar, para médias tirar média
	agg_funcs = {}
	for c in meas_cols:
		if 'PRECIPITACAO TOTAL' in c or 'NUMERO DE DIAS' in c:
			agg_funcs[c] = 'sum'
		else:
			agg_funcs[c] = 'mean'

	# Agrupar por MesAno e Regiao
	temp_agg = df.groupby(['MesAno', 'Regiao'], dropna=False).agg(agg_funcs).reset_index()
	return temp_agg


def join_consumo_temperatura(consumo_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
	# Left join consumo com temperatura por MesAno e Regiao
	joined = consumo_df.merge(temp_df, on=['MesAno', 'Regiao'], how='left')

	# adicionar coluna Estacao a partir de MesAno (usar mês de MesAno)
	meses = pd.to_datetime(joined['MesAno'] + '-01')
	def mes_para_estacao(m):
		m = m.month
		if m in (12, 1, 2):
			return 'verão'
		if m in (3, 4, 5):
			return 'outono'
		if m in (6, 7, 8):
			return 'inverno'
		return 'primavera'

	joined['Estacao'] = meses.map(mes_para_estacao)
	return joined


if __name__ == '__main__':
	base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	consumo_path = os.path.join(base, 'consumo_energia', 'Dados_residencial.csv')
	temp_path = os.path.join(base, 'temperatura', 'temperatura_mensal_com_regiao_final.csv')
	out_path = os.path.join(base, 'manipulação', 'consumo_vs_temperatura_por_regiao.csv')

	print('Agregando consumo...')
	consumo = aggregate_consumption(consumo_path)
	print('Agregando temperatura...')
	temp = aggregate_temperature(temp_path)
	print('Juntando...')
	result = join_consumo_temperatura(consumo, temp)
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	result.to_csv(out_path, index=False, sep=';', decimal=',')
	print(f'Salvo em: {out_path} — linhas: {len(result)}')

