import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    df_binary = df[['Salnty', 'T_degC']]
    

    df_binary.columns = ['Sal', 'Temp']

    df_binary.head()
    print(df)
