import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import streamlit as st

st.header(':green[Linear] :blue[Regression] :red[Practice!!]')
uploaded_file = st.file_uploader("Choose a file", type=[".csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sizeoftest=st.number_input(
        "Select _test size_",
        min_value=0.0,
        max_value=1.0,
    )
    input1 = st.multiselect("Select  _your feature_", list(df.columns.values)[:-1])
    X=df[input1].values
    y= df.iloc[:,-1].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,y,test_size=sizeoftest,random_state=40
        )
    st.markdown("**_Your uploaded data:_** ")
    st.write(df)
    lr=LinearRegression()
    lr.fit(X_train, Y_train)
    Y_pred=lr.predict(X_test)
    predict1=st.button('Predict')
    visualize=st.button('Visualize')
    if predict1:
        st.write(Y_pred)
    if visualize:
        p=plt.figure(figsize=(12,4))
        ax=p.add_subplot(121)
        sns.distplot(df.iloc[:,-1].values,color='g',hist=True)
        ax.set_title("Visualize Distribution of "+df.columns.values[-1])
        st.pyplot(p)
        fig,axx=plt.subplots()
        axx.scatter(X_train,Y_train,color='red')
        axx.plot(X_train,lr.predict(X_train),color='blue')
        axx.set_title("Visualize Training Data Set")
        st.pyplot(fig)
        fig1,axx1=plt.subplots()
        axx1.scatter(X_test,Y_test,color='purple')
        axx1.plot(X_test,lr.predict(X_test),color='yellow')
        axx1.set_title("Visualize Testing Data Set")
        st.pyplot(fig1)
        
        

            
                
            
            
            
                
        
        
    
    
    
    
    







