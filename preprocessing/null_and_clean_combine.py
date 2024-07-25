
import pandas as pd

xls = pd.ExcelFile("/content/data/230624_all_data_workup.xlsx")

df = pd.read_excel(xls, 'Normalized intensities and peak')

seqs = df.Sequence.to_frame()

fpath  = "/content/data/supercleanGMMFiltered.xlsx"

xls = pd.ExcelFile(fpath)
data = pd.read_excel(xls)

mDF = seqs.merge(data,how='left',on='Sequence')

mDF.to_csv('combined_null_and_dis.csv')


import collections
print([item for item, count in collections.Counter(mDF.Sequence).items() if count > 1])

mDF.loc[  mDF.Sequence == 'TTTCGTCTCC']


#print the column names
print(df.columns)

print("end")
