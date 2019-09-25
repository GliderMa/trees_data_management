import pandas as pd
Variable=['Export','FDI','GEE','HTE','Import','PD','UPT']
df=pd.read_csv('sum.csv')
def get_country_value(dataframe,country,year):
    a=dataframe.loc[dataframe['Country Code']==country]
    try:
        result=a.iloc[0][year]
    except:
        result=None
    return result

for var in Variable:
    var_file=var+'.csv'
    df_var=pd.read_csv(var_file, encoding='cp1252')
    df[var]=None
    for record in range(0, len(df), 1):
        key = df.at[record, 'Key']
        country = df.at[record, 'Code']
        year = str(df.at[record, 'Year'])
        value=get_country_value(df_var,country,year)
        df.at[record,var]=value

df.to_csv('test.csv',index=None)
'''
df1=pd.read_csv('PD.csv',encoding='cp1252')
a=get_country_value(df1,'AGO','1993')

print(a)
'''