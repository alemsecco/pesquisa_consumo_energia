import pandas as pd
import unicodedata

# --- 1. FUNÇÕES DE NORMALIZAÇÃO E LIMPEZA ---
def normalize_text(s):
    if not isinstance(s, str): return s
    s = s.upper()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = ''.join(e for e in s if e.isalnum() or e.isspace())
    return s.strip()

def clean_city_name(city_name):
    if not isinstance(city_name, str): return city_name
    return city_name.split(' - ')[0].strip()

# --- 2. EXECUÇÃO PRINCIPAL ---
try:
    # --- ETAPA A: BAIXAR E PREPARAR A LISTA OFICIAL DE MUNICÍPIOS ---
    print("Baixando a lista oficial de municípios do Brasil...")
    url_municipios = 'https://www.gov.br/receitafederal/dados/municipios.csv'
    df_municipios = pd.read_csv(url_municipios, sep=';', encoding='latin1')
    df_municipios = df_municipios[['MUNICÍPIO - IBGE', 'UF']]
    df_municipios.columns = ['Cidade', 'UF']
    df_municipios['Cidade_norm'] = df_municipios['Cidade'].apply(normalize_text)
    df_municipios_unicos = df_municipios.drop_duplicates(subset=['Cidade_norm'])
    city_to_uf_map = pd.Series(df_municipios_unicos.UF.values, index=df_municipios_unicos.Cidade_norm).to_dict()
    print(f"Mapeamento com {len(city_to_uf_map)} municípios únicos criado.")

    # --- ETAPA B: CARREGAR E PROCESSAR O SEU ARQUIVO ---
    print("\nCarregando seu arquivo 'temperatura_mensal_unificado.csv'...")
    caminho_do_arquivo = r'dados\temperatura\temperatura_mensal_unificado.csv'
    df_temp = pd.read_csv(caminho_do_arquivo, delimiter=';')

    # --- ETAPA C: PREENCHER AS UFs AUSENTES USANDO O MAPEAMENTO ---
    print("Preenchendo as UFs ausentes...")
    # Cria colunas temporárias para a correspondência
    df_temp['Cidade_limpa'] = df_temp['Cidade'].apply(clean_city_name)
    df_temp['Cidade_norm'] = df_temp['Cidade_limpa'].apply(normalize_text)
    
    # Preenche a coluna UF. Usamos .copy() para evitar avisos.
    df_temp_filled = df_temp.copy()
    df_temp_filled['UF'].fillna(df_temp_filled['Cidade_norm'].map(city_to_uf_map), inplace=True)

    # --- ETAPA D: MAPEAR UF PARA REGIÃO ---
    uf_to_region = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }
    df_temp_filled['Regiao'] = df_temp_filled['UF'].map(uf_to_region)
    
    # --- ETAPA E: SALVAR O ARQUIVO COMPLETO ---
    df_final = df_temp_filled.drop(columns=['Cidade_limpa', 'Cidade_norm'])
    output_filename_completo = 'temperatura_mensal_com_regiao_final.csv'
    df_final.to_csv(output_filename_completo, sep=';', index=False)
    print(f"\nArquivo completo salvo como '{output_filename_completo}'")

    # --- ETAPA F (NOVO): SELECIONAR E SALVAR APENAS AS UFs NÃO ENCONTRADAS ---
    print("-" * 50)
    print("\nVerificando se alguma UF não foi encontrada...")
    
    # Filtra o dataframe final para pegar apenas as linhas onde 'UF' é nulo
    df_nao_encontradas = df_final[df_final['UF'].isnull()]

    if not df_nao_encontradas.empty:
        print(f"Foram encontradas {len(df_nao_encontradas)} linhas com UFs não identificadas.")
        
        # Mostra os nomes únicos das cidades problemáticas
        cidades_problematicas = df_nao_encontradas['Cidade'].unique()
        print("\nCidades que não foram encontradas no mapa oficial:")
        for cidade in cidades_problematicas:
            print(f"- {cidade}")
            
        # Salva essas linhas em um arquivo separado
        output_filename_nao_encontradas = 'temperatura_ufs_nao_encontradas.csv'
        df_nao_encontradas.to_csv(output_filename_nao_encontradas, sep=';', index=False)
        print(f"\nUm arquivo com essas linhas foi salvo como '{output_filename_nao_encontradas}' para sua análise.")
    else:
        print("Ótima notícia! Todas as UFs foram preenchidas com sucesso.")
        
except FileNotFoundError:
    print("\nERRO: O arquivo 'temperatura_mensal_unificado.csv' não foi encontrado.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")