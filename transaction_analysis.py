import duckdb
conn = duckdb.connect('aml.duckdb')
# print(conn.execute('DROP TABLE IF EXISTS transactions').df())
conn.execute('''
CREATE TABLE IF NOT EXISTS transactions AS Select * FROM
    read_csv_auto('HI-Small_trans_cleaned.csv',Header=True)
    ''')



###########LOGIC TO CLEAN THE COLUMN NAMES AND WRITE BACK TO CSV FILE###########
# import pandas as pd
# df=pd.read_csv('HI-Small_trans_cleaned.csv')
# columns=df.columns.tolist()

# columns[2]="from_account"
# columns[4]="to_account"

# new_columns=[]
# #fix the code for column names to be lower case and replace spaces with underscores
# for col in columns:
#     col=col.replace(" ","_").lower()    
#     new_columns.append(col)

# df.columns=new_columns

# for col in df.columns:
#     print(col)


# #write the new columns to the existing csv file
# df.to_csv('HI-Small_trans_cleaned.csv',index=False)
################

#print(conn.execute('SELECT * FROM transactions LIMIT 5').df().to_markdown())


print(conn.execute('SELECT count(*) FILTER(WHERE is_laundering=1) *100  / count(*) as total_count FROM transactions ').df().to_markdown())
print(conn.execute('SELECT count(*) FILTER(WHERE is_laundering=0) *100  / count(*) as total_count FROM transactions ').df().to_markdown())