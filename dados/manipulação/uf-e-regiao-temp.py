import pandas as pd
import unicodedata

# --- 1. FUNÇÕES DE NORMALIZAÇÃO E LIMPEZA ---
def normalize_text(s):
    """
    Remove acentos, converte para maiúsculas e padroniza o texto.
    """
    if not isinstance(s, str):
        return s
    # Converte para maiúsculas
    s = s.upper()
    # Remove acentos
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # Remove caracteres especiais, mantendo apenas letras e espaços
    s = ''.join(e for e in s if e.isalnum() or e.isspace())
    # Remove espaços extras no início e no fim
    s = s.strip()
    return s

def clean_city_name(city_name):
    """
    Limpeza específica para os nomes de cidade do seu arquivo, 
    removendo sufixos (ex: ' - CERCADINHO').
    """
    if not isinstance(city_name, str):
        return city_name
    return city_name.split(' - ')[0].strip()

# --- 2. EXECUÇÃO PRINCIPAL ---
try:
    # --- ETAPA A: BAIXAR E PREPARAR A LISTA OFICIAL DE MUNICÍPIOS ---
    print("Baixando a lista oficial de municípios do Brasil...")
    url_municipios = 'https://www.gov.br/receitafederal/dados/municipios.csv'
    # Use encoding 'latin1' que é comum para arquivos de governo no Brasil
    df_municipios = pd.read_csv(url_municipios, sep=';', encoding='latin1')
    
    # Selecionar e renomear as colunas de interesse
    df_municipios = df_municipios[['MUNICÍPIO - IBGE', 'UF']]
    df_municipios.columns = ['Cidade', 'UF']
    
    # Normalizar os nomes das cidades na lista oficial para criar uma chave de busca
    df_municipios['Cidade_norm'] = df_municipios['Cidade'].apply(normalize_text)
    
    # Criar um dicionário de mapeamento: {nome_normalizado: UF}
    # Remove duplicatas para garantir um mapeamento único
    df_municipios_unicos = df_municipios.drop_duplicates(subset=['Cidade_norm'])
    city_to_uf_map = pd.Series(df_municipios_unicos.UF.values, index=df_municipios_unicos.Cidade_norm).to_dict()
    print(f"Mapeamento com {len(city_to_uf_map)} municípios únicos criado com sucesso.")

    # --- ETAPA B: CARREGAR E PROCESSAR O SEU ARQUIVO ---
    print("\nCarregando seu arquivo 'temperatura_mensal_unificado.csv'...")
    # **IMPORTANTE**: Certifique-se que o seu arquivo CSV está na mesma pasta que este script
    caminho_do_arquivo = r'dados\temperatura\temperatura_mensal_unificado.csv'
    df_temp = pd.read_csv(caminho_do_arquivo, delimiter=';')

    ufs_ausentes_antes = df_temp['UF'].isnull().sum()
    print(f"Seu arquivo tem {ufs_ausentes_antes} linhas com a UF faltando.")

    # Limpar e normalizar a coluna de cidades do seu arquivo
    df_temp['Cidade_limpa'] = df_temp['Cidade'].apply(clean_city_name)
    df_temp['Cidade_norm'] = df_temp['Cidade_limpa'].apply(normalize_text)

    # --- ETAPA C: PREENCHER AS UFs AUSENTES USANDO O MAPEAMENTO ---
    print("Preenchendo as UFs ausentes...")
    df_temp['UF'].fillna(df_temp['Cidade_norm'].map(city_to_uf_map), inplace=True)

    ufs_ausentes_depois = df_temp['UF'].isnull().sum()
    print(f"{ufs_ausentes_antes - ufs_ausentes_depois} UFs foram preenchidas automaticamente.")
    if ufs_ausentes_depois > 0:
        print(f"ATENÇÃO: {ufs_ausentes_depois} UFs não foram encontradas no mapa oficial.")


    # --- ETAPA D: AJUSTES MANUAIS ---
    print("\nAplicando ajustes manuais definidos por você...")
    
    # ** Adicione suas correções aqui **
    # Formato: "NOME EXATO DA CIDADE NO ARQUIVO": "UF CORRETA"
    correcoes_manuais = {
        "BOM JESUS DO PIAUI": "PI",
        "FACULDADE DA TERRA DE BRASILIA": "DF",
        "MARIANOPOLIS DO TO": "TO",
        "BRAZLANDIA": "DF",
        "AGUAS EMENDADAS": "DF",
        "GAMA (PONTE ALTA)": "DF",
        "PARANOA (COOPA-DF)": "DF",
        "CRISTALINA (FAZENDA SANTA MONICA)": "GO",
        "PARQUE ESTADUAL CHANDLESS": "AC",
        "RIO URUBU": "AM",
        "S. G. DA CACHOEIRA": "AM",
        "TOME ACU": "PA",
        "FAROL de SANTANA": "MA",
        "PREGUICAS": "MA",
        "SERRA DOS CARAJAS": "PA",
        "MINA DO PALITO": "PA",
        "ARQ.SAO PEDRO E SAO PAULO": "PE",
        "ARCO VERDE": "PE",
        "CALCANHAR": "RN",
        "CAMARATUBA": "PB",
        "AREMBEPE": "BA",
        "LUIZ EDUARDO MAGALHAES": "BA",
        "ABROLHOS": "BA",
        "DELFINO": "BA",
        "SALVADOR (RADIO FAROL)": "BA",
        "MONTE VERDE": "MG",
        "MOCAMBINHO": "PI",
        "IBIRITE (ROLA MOCA)": "MG",
        "NOVA PORTEIRINHA (JANAUBA)": "MG",
        "SEROPEDICA-ECOLOGIA AGRICOLA": "RJ",
        "RIO DE JANEIRO-MARAMBAIA": "RJ",
        "PICO DO COUTO": "RJ",
        "TERESOPOLIS-PARQUE NACIONAL": "RJ",
        "ILHA DE TRINDADE": "ES",
        "ESCOLA NAVAL": "RJ",
        "LAGOA R.DE FREITAS": "RJ",
        "JACAREPAGUA": "RJ",
        "ENGENHO DE DENTRO": "RJ",
        "NHUMIRIM": "SP",
        "MOELA": "SP",
        "MAL. CANDIDO RONDON": "PR",
        "ILHA DO MEL": "PR",
        "S.J. DO RIO CLARO": "MT",
        "CAMPO NOVO DOS PARECIS": "MT",
        "BRASNORTE (NOVO MUNDO)": "MT",
        "PORTO ALEGRE- BELEM NOVO": "RS",
        "PARQUE ELDORADO": "RS"
    }
    
    for cidade, uf_correta in correcoes_manuais.items():
        # A lógica é: onde a coluna 'Cidade' for igual à chave do dicionário,
        # preencha a coluna 'UF' com o valor correspondente.
        df_temp.loc[df_temp['Cidade'] == cidade, 'UF'] = uf_correta

    ufs_nao_encontradas_final = df_temp['UF'].isnull().sum()
    print(f"Após os ajustes manuais, restam {ufs_nao_encontradas_final} linhas sem UF.")


    # --- ETAPA E: MAPEAR UF PARA REGIÃO ---
    print("\nMapeando UFs para as Regiões correspondentes...")
    uf_to_region = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }
    df_temp['Regiao'] = df_temp['UF'].map(uf_to_region)

    # --- ETAPA F: SALVAR O ARQUIVO FINAL ---
    # Limpar colunas auxiliares
    df_temp.drop(columns=['Cidade_limpa', 'Cidade_norm'], inplace=True)

    df_temp.rename(columns={'Cidade': 'Local'}, inplace=True)
    
    output_filename = 'temperatura_mensal_com_regiao_final.csv'
    df_temp.to_csv(output_filename, sep=';', index=False)
    
    print(f"\nPROCESSO CONCLUÍDO! O arquivo final foi salvo como '{output_filename}'")

except FileNotFoundError:
    print("\n--- ERRO ---")
    print("O arquivo 'temperatura_mensal_unificado.csv' não foi encontrado.")
    print("Por favor, certifique-se de que ele está na mesma pasta onde você está executando este script.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")