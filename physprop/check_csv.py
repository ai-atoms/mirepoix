import pandas as pd
import ast  # to safely parse stringified lists

df = pd.read_csv('data/datasets/d567b/pcomp_d567b.csv', encoding="utf-7")

# Minor adjusts
df['C15'] = df['C15'].str.replace(r'[\[\]\s]+', ',', regex=True).str.strip(',').apply(lambda x: '[' + x + ']')
print (df['C15'].tail())

# Safely parse each entry into a list and compute its length
max_len = df['C15'].apply(lambda x: len(ast.literal_eval(x))).max()
print("Maximum vector length in C15:", max_len)