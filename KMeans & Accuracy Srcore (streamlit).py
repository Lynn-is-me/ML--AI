import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn import neighbors,datasets
from sklearn.metrics import silhouette_score, accuracy_score, mean_squared_error
import streamlit as st


st.title(':green[Clustering] :blue[K- means] :red[Practice!!]')
uploaded_file = st.file_uploader("Choose a file", type=[".csv"])
if uploaded_file is not None:
    #data
    df = pd.read_csv(uploaded_file)
    st.markdown("**_Your uploaded data:_** ")
    st.write(df)
    X = df.copy()
    input1 = []
    input1 = st.multiselect("Select  _your feature_", list(df.columns.values))
    n_features = np.genfromtxt(input1, delimiter=',', dtype=str)
    X_scaled = X.loc[:, n_features]
    
    #elbow method
    kvalue = []
    wcss = []
    for i in range(2,10):
        kmeans=KMeans(n_clusters=i)
        y_pred=kmeans.fit(X_scaled)
        kvalue.append(i)
        wcss.append(kmeans.inertia_)     
    k = pd.DataFrame({'k':kvalue, 
                      'wcss': wcss})
    st.write('# Elbow Method')
    st.line_chart(k,x='k',y='wcss')
    
    #users'input
    n_clusters1 = st.number_input(
        '_the number of clusters_: ',
        min_value=1,
        max_value=10,
        )
    init1=st.selectbox(
        '_init parameter_',
        ('random','k-means++')
    )
    n_init1=st.number_input(
        '_n_init parameter_ (0 for auto)',
        min_value=0,
        max_value=10,
    )
    if n_init1==0: n_init1 = 'auto'
    
    #kmeans
    kmeans1 = KMeans(n_clusters=n_clusters1, init=init1, n_init=n_init1,random_state=0 )
    X["Clusters"] = kmeans1.fit_predict(X_scaled)
    st.write(X["Clusters"]) #X

    
    #visualize kmeans result
    X["Clusters"] = X["Clusters"].astype("category")
    fig = plt.figure(figsize=(10,4))
    sns.set_style('whitegrid')
    
    choose_x=st.selectbox(
        '_Select x_',
        list(n_features),
    )
    choose_y=st.selectbox(
        '_Select y_',
        list(n_features),
    )
    fig = sns.relplot(x= choose_x, y= choose_y, hue=X['Clusters'], data=X_scaled, kind='scatter')
    st.pyplot(fig)
    
    #training
    sizetest = st.number_input(
        "Choose _test size_: ",
        min_value=0.1,
        max_value=1.0,
    )
    
        #data
    y = X.iloc[:,-1].values
    X = X.loc[:,n_features].values #X
    yy = y.to_numpy()
    
        #classification
    clf = LinearRegression()
    X_train,X_test,Y_train,Y_test = train_test_split(X,yy,test_size=sizetest,random_state=45)  
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    st.write(
        "_Predicted labels:_ ",
        Y_pred,
    )
    st.write(
        "_Ground truth:_ ",
        Y_test,
    )
        #evaluation
    result = mean_squared_error(Y_test,Y_pred)
    st.write(
        "Accuracy score: ",
        result,
    )
    
    
    
    
    
    
    
    
    
    
   