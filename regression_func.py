
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data(filename):
    df = pd.read_csv(filename)
    df_binary = df[['Salnty', 'T_degC']]
    
    df_binary.columns = ['Sal', 'Temp']

    df_binary.head()
    print(df)
    
if __name__=="__main__":
    load_data("bottle.csv")
    
