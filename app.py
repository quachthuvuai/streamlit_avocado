import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pmdarima.arima import ARIMA
from pmdarima import auto_arima

st.write("""
# Business Objective
Hass, a company based in Mexico, specializes in producing a variety of avocados for selling in U.S.A. 
They have been very successful over recent years and want to expand their business. 
Thus they want build a reasonable model to predict the average price of avocado “Hass” in the U.S.A to consider the expansion of existing avocado farms in other regions.

There are two types of avocados (conventional and organic) in the dataset and several different regions. 
This allows us to do analysis for either conventional or organic avocados in different regions and/or the entire United States.

There are 2 different approaches to solve this business objective:
First approach: create a regression model using supervised machine learning algorithms such as Linear Regression, Random Forest, XGB Regressor so on to predict average price of avocado in the USA.
Second approach: build a predictive model based on supervised time-series machine learning algorithms like Arima, Prophet, HoltWinters to predict average price of a particular avocado (organic or conventional) over time for a specific region in the USA.
"""
)

# Images
from PIL import Image 
img = Image.open("avocado_1.jpg")
st.image(img,width=700,caption='Streamlit Images')


st.write("""
# Avocado price Forecasting using ARIMA model
This app used avocado.csv dataset as train and test data. 
""")

st.info('Objective: To predict / forecast the average price of Avocado based on time series data using ARIMA model')


# Images
# from PIL import Image 
# img = Image.open("avocado_2.jpg")
# st.image(img,width=700,caption='Streamlit Images')


# Reads test data
organic_data = pd.read_csv('organic_test.csv')
organic_test=organic_data.set_index(pd.DatetimeIndex(organic_data['Date'])).drop('Date', axis=1)
# Reads test data
conventional_data = pd.read_csv('organic_test.csv')
conventional_test=conventional_data.set_index(pd.DatetimeIndex(conventional_data['Date'])).drop('Date', axis=1)


# Reads in saved arima model
organic_model = pickle.load(open('arima_model_organic.pkl', 'rb'))
conventional_model = pickle.load(open('arima_model_conventional.pkl', 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Avocado Type',
                          
                          ['organic',
                           'conventional'],
                          icons=['heart','person'],
                          default_index=0)
    
    
    # number of month to predict
    no_months = st.sidebar.slider('Number of months to predict', 1, 48, 12)

    # Information about us
    st.sidebar.title("About us")
    st.sidebar.info(
        """
        This web [app](....) is maintained by [Quách Thu Vũ & Thái Văn Đức]. 
        Học Viên lớp LDS0_K279 | THTH DHKHTN |
    """
    )


def run():

    # number of month to predict
    #no_months = st.sidebar.slider('Number of months to predict', 1, 48, 12)

    if st.button('Run'):
        if (selected == 'organic'):
            # Apply model to make predictions
            prediction = organic_model.predict(n_periods=len(organic_test)+no_months)
            st.success(f'Price prediction for the next {no_months} months is sucessful')
            st.dataframe(prediction)
            last_date=prediction.last_valid_index()
            last_price=prediction.iloc[-1]
            st.success(f'Organic Avocado price at {last_date} is {last_price:.2f} USD')

            fig = plt.figure()
            plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(prediction, label='Prediction price')
            ax.legend()
            st.pyplot(fig)
        else:
            # Apply model to make predictions
            prediction = conventional_model.predict(n_periods=len(conventional_test)+no_months)
            st.success(f'Price prediction for {no_months} months is sucessful')
            st.dataframe(prediction)
            last_date=prediction.last_valid_index()
            last_price=prediction.iloc[-1]
            st.success(f'Conventional Avocado price at {last_date} is {last_price:.2f} USD')

            fig = plt.figure()
            plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(prediction, label='Prediction price')
            ax.legend()
            st.pyplot(fig)
    else:
        st.error('Please click "Run" to predict the future price!')
run()