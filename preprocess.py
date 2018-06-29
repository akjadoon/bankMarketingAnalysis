##### Reads in the input CSV Data
##### Converts column values which are strings to numbers

import pandas as pd 


DATA_FILE = "bank-additional-full.csv"
RESULT_FILE= "procd-" + DATA_FILE

df = pd.read_csv(DATA_FILE, delimiter=";")

###Reorder the columns

###Get only columns where data is not a number
str_cols = list(df.select_dtypes(include='object'))
all_cols = df.columns.tolist()

#Get unique values and map to an int
for col in str_cols:
    col_values = list(df[col].unique())
    df[col] = df[col].apply(lambda str: col_values.index(str))

###Write result to csv
df[str_cols].to_csv("cat"+RESULT_FILE, index=False)
df[list(set(all_cols)-set(str_cols))].to_csv("num"+RESULT_FILE, index=False)