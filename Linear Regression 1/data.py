import pandas as pd
import streamlit as st
def load_data(filename):
    df = pd.read_csv(filename)
    df_binary = df[['Salnty', 'T_degC']]
    
    # Taking only the selected two attributes from the dataset
    df_binary.columns = ['Sal', 'Temp']
    #display the first 5 rows
    df_binary.head()
    print(df)
    
    