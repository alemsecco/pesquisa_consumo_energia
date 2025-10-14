# 🔎 Previsão de Consumo Energético em Residências (casas e edifícios)

## 📝 Resumo da pesquisa
Em um contexto de cidades inteligentes, desenvolvimento sustentável e demanda energética, torna-se urgente utilizar métodos atuais para adaptar nosso consumo de energia.

A partir disso, o objetivo dessa pesquisa é desenvolver um modelo de _machine learning_ capaz de prever o consumo energético de residências e edifícios e, a partir desses dados, recomendar a alternativa sustentável de energia mais adequada ao cliente, de forma a contribuir com soluções sustentáveis e com o avanço da pesquisa em ciência da computação.

## 💻 Tecnologias utilizadas
- _Machine Learning_
- IoT
- _Big Data_
- _Business Intelligence_
- Python 3.12.9

## 📈 Dados
### ⚡Energia
Os dados utilizados neste projeto referem-se ao consumo mensal de energia elétrica no Brasil e foram obtidos através do portal de Dados Abertos da Empresa de Pesquisa Energética (EPE).

- **Fonte:** Empresa de Pesquisa Energética (EPE)
- **Dataset:** Consumo Mensal de Energia Elétrica
- **Link para os dados:** [https://www.epe.gov.br/pt/publicacoes-dados-abertos/dados-abertos/dados-do-consumo-mensal-de-energia-eletrica](https://www.epe.gov.br/pt/publicacoes-dados-abertos/dados-abertos/dados-do-consumo-mensal-de-energia-eletrica)
- **Data de acesso:** 10/10/2025

### 🌡️ Temperatura
Para obter dados mensais de temperatura, pressão atmosférica, velocidade do vento e precipitação, utilizamos os dados do Instituro Nacional de Meteorlogia (INMET), via solicitação por e-mail de dados de 2004 a 2025.

- **Fonte:** Instituto Nacional de Meteorologia (INMET)
- **Link para solicitação de dados:** [https://bdmep.inmet.gov.br/](https://bdmep.inmet.gov.br/)
- **Data de acesso:** 10/10/2025

### 👥 População
Os dados referentes à população brasileira foram retirados do Instituto Brasileiro de Geografia e Estatística (IBGE). Os dados sobre a população atual referem-se ao Censo de 2022, enquanto os dados de crescimento populacional foram retirados do panorama do Censo.

- **Fonte:** Instituto Brasileiro de Geografia e Estatística (IBGE)
- **Link para dados dos censos:** [https://www.ibge.gov.br/estatisticas/sociais/saude/22827-censo-demografico-2022.html?=&t=downloads](https://www.ibge.gov.br/estatisticas/sociais/saude/22827-censo-demografico-2022.html?=&t=downloads)
- **Link para dados do panorama:** [https://censo2022.ibge.gov.br/panorama/](https://censo2022.ibge.gov.br/panorama/)
- **Data de acesso:** 10/10/2025

## 📊 Manipulação dos dados
Primeiramente, [filtramos](dados/manipulação/energia.py) os dados da EPE para obter somente os consumidores residenciais, que são o foco do projeto. Os dados filtrados estão disponíveis na tabela [Dados_residencial.csv](dados/consumo_energia/Dados_residencial.csv). Após isso, adicionamos mais uma coluna a essa tabela para termos os dados das estações do ano em que cada medida foi feita, para podermos analisar a influência da estação no consumo energético.

O próximo passo foi [reunir todos os arquivos](dados/manipulação/unificacao-temp.py) proporcionados pelo INMET em uma única tabela, e [adicionamos as UFs e regiões](dados/manipulação/uf-e-regiao-temp.py) correspondentes para cada local. A tabela completa pode ser acessada por [aqui](dados/temperatura/temperatura_mensal_com_regiao_final.csv).

Finalmente, cruzamos os dados das duas tabelas, gerando uma terceira tabela que indica o mês e ano, estação (verão, inverno, outono, primavera), região (Norte, Sul, Nordeste, Sudeste, Centro-Oeste), consumo de energia e os dados de temperatura, pressão, precipitação e vento. Esses dados estão disponíveis em [consumo_vs_temperatura_por_regiao](dados/consumo_vs_temperatura_por_regiao.csv).

## ✍️ Autores
Copyright © 2025 Alex Menegatti Secco e Mariana de Castro
