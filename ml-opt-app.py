#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:01:12 2021

@author: kevingarcia
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import requests, io
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from sklearn.model_selection import learning_curve


#---------------------------------#
# Page layout
## Page expands to full width    
# Start clock
start = timeit.default_timer()
st.set_page_config(page_title = 'Kevin\'s Data Report Application',
    layout='wide')


       

#---------------------------------#
st.write("""
# Kevin\'s Data Report Application
In this implementation, the *RandomForestClassifier()* function is used in this app to build a 
classification model using the **Random Forest** algorithm. When inputing a csv file, please use a 
classification problem. Regression analysis **Coming Soon**.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example Dataset](https://raw.githubusercontent.com/kgarcia07/Data_Demos/main/heart.csv)
""")

# Sidebar - Specify parameter settings
st.sidebar.write('---')
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
st.sidebar.write('---')

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
st.sidebar.write('---')
parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1,3), 1)
#st.sidebar.number_input('Step size for max_features', 1)
#st.sidebar.write('---')
#parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
#parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
#parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
#parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
#parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
#parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

#---------------------------------#
# Main panel
    
# Displays the dataset
st.subheader('Dataset')

#---------------------------------#
# Model building

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    #X_train.shape, Y_train.shape
    #X_test.shape, Y_test.shape
    
    #rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        #random_state=parameter_random_state,
        #max_features=parameter_max_features,
        #criterion=parameter_criterion,
        #min_samples_split=parameter_min_samples_split,
        #min_samples_leaf=parameter_min_samples_leaf,
        #bootstrap=parameter_bootstrap,
        #oob_score=parameter_oob_score,
        #n_jobs=parameter_n_jobs)
    
    rf = RandomForestClassifier(n_estimators=parameter_n_estimators,max_features=parameter_max_features,n_jobs=parameter_n_jobs)
    #grid = rf
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)
    
    
    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)   
    
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )
    
    test_score = round(grid.score(X_test, Y_test) * 100, 2)
    st.write('Testing Score:')
    st.info( test_score )
    
    train_score = round(grid.score(X_train, Y_train) * 100, 2)
    st.write('Training Score:')
    st.info( train_score )
        
    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )
    
    st.write("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    
    
    #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    st.write('Hyperparameter tuning')
    layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='n_estimators')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_features')
            ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(scene = dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)
    
    #Correlation Matrix Heat Map
    st.write('Correlation Matrix Heat Map')
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    #fig.update_layout(title='Correlation Matrix Heat Map')
    st.write(fig)
    
    st.write('Proportion of our Target Lables')
    chart = plt.figure(figsize=(10,6),facecolor = '#90eef8')
    plt.pie(df.iloc[:,-1].value_counts(),labels = np.unique(df.iloc[:,-1]),startangle=90,explode=[0,0.05],pctdistance=0.8,shadow=True,colors=['#fdb98a','#f76502'],autopct = '%1.1f%%')
    centre_circle = plt.Circle((0,0),0.60,fc='#90eef8')
    plt.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    st.write(chart)
    
    
    train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X_train, Y_train, cv = 10, scoring = 'accuracy', n_jobs = -1,
                                                                train_sizes=np.linspace(0.01, 1, 50))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    st.write('Learning Curve')
    fig, ax = plt.subplots(1, figsize=(10,10))
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
        
    #plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    st.write(fig)
    
    #-----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x,y,z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)

#---------------------------------#
url = "https://raw.githubusercontent.com/kgarcia07/Data_Demos/main/heart.csv"
request = requests.get(url).content
example_data = pd.read_csv(io.StringIO(request.decode('utf-8')))
      
                             
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(10))
    build_model(df)
    # All the program statements
    stop = timeit.default_timer()
    execution_time = stop - start
    st.text("Program Executed in "+str(execution_time) + " Seconds")
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        df = example_data
        st.markdown('The **HEART** dataset is used as the example. Showing first 10 rows:')
        st.write(df.head(10))
        build_model(df)
        # All the program statements
        stop = timeit.default_timer()
        execution_time = stop - start
        st.text("Program Executed in "+str(execution_time) + " Seconds")
        








