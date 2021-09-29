import pandas as pd

file_name = 'BW16x16.xlsx'
df = pd.read_excel(r'./BW16x16.xlsx')
print(df)
print(df.dtypes)

