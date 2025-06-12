import pandas as pd
import ast

# Caricamento del dataframe (supponendo che sia già disponibile come df)
df = pd.read_csv("single_results_all_confs.csv")

# Converte la colonna DICE da stringa a lista
df['DICE'] = df['DICE'].apply(ast.literal_eval)

# Aggrega i valori per Setting e Method
grouped = df.groupby(['Setting', 'Method'])['DICE'].sum()

# Calcola la media per ciascun gruppo
mean_dice = grouped.apply(lambda x: sum(x) / len(x))

# Visualizza il risultato
print(mean_dice)