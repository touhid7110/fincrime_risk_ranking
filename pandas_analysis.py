import pandas as pd
df = pd.read_csv('Hi-Small_Trans.csv')
cols = df.columns.tolist()
cols[2]="from_account"
cols[4]="to_account"
for i in range(len(cols)):
    cols[i] = cols[i].replace(" ", "_").lower()
df.columns = cols
df.to_csv("HI-Small_Trans_cleaned.csv", index=False)
