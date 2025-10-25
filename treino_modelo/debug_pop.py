from treino_modelo.train_global_and_regions import load_population_by_region, add_population_features
import treino_modelo.train_global_and_regions as tgr
pop = load_population_by_region()
print('pop_map:', pop)

from treino_modelo import treino
df = treino.load_data()
df['Consumo'] = df['Consumo'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
df['Consumo'] = pd.to_numeric(df['Consumo'], errors='coerce')
df2 = add_population_features(df)
print(df2[['Regiao','Pop_2022','Consumo','Consumo_per_1000hab']].head(10).to_string())
