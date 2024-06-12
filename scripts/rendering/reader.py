import csv
import pandas as pd
import ast

# File path
filename = '/fs/nexus-scratch/sjxu/objaverse-xl/scripts/rendering/meta_unique.txt'  # Replace with your text file name

# Read the file into a pandas DataFrame
df = pd.read_csv(filename, sep='\t')

for index, row in df.iterrows():
    # print(row['color'])
    # print(ast.literal_eval(row['color']))
    a = ast.literal_eval(row['color'])
    print(a)
    # print(tuple((row['color'])))
    # print(type(row['color']) is tuple)
