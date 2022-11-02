import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot

from pmdarima.arima import ARIMA
from pmdarima import auto_arima

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import LabelEncoder
import pickle
import json
# from prophet.serialize import model_to_json, model_from_json


# Page setting
st.set_page_config(layout="wide")


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Wellcome to our App',
                          
                          ['Introduction',
                          'Data Exploration',
                          'Regression model',
                           'Arima model',
                           'Prophet model',
                           'Evaluation',
                           'Thank You'],
                          icons=['book', 'key', 'moon', 'calculator', 'pen', 'sun', 'person'],
                          default_index=0)
    
    


    # Sidebar setup
    st.sidebar.title(':arrow_up: Upload data here:')
    uploaded_file = st.sidebar.file_uploader('Upload avocado data', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.to_csv("avocado_new.csv", index = False)

    # Information about us
    st.sidebar.title(":two_men_holding_hands: About us")
    st.sidebar.info(
        """
        This web [app](....) is maintained by [Quách Thu Vũ & Thái Văn Đức]. 
        Học Viên lớp LDS0_K279 | THTH DHKHTN |
    """
    )

    st.markdown('This dashboard is made by **Streamlit**')
    st.sidebar.image('https://streamlit.io/images/brand/streamlit-mark-color.png', width=50)

#=====================================================================================================================================================

@st.cache
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    return df

data = load_data('avocado.csv')
# sort by Date
data=data.sort_values("Date")
data.drop(['Unnamed: 0'], axis=1, inplace=True)
avocado_stats = data.groupby('type')['AveragePrice', 'Total Volume', 'Total Bags'].mean()


data['Date']=pd.to_datetime(data['Date'])
data['week']=pd.DatetimeIndex(data['Date']).week
data['month']=pd.DatetimeIndex(data['Date']).month

def convert_month(month):
  if month==3 or month==4 or month==5:
    return 0
  elif month==6 or month==7 or month==8:
    return 1
  elif month==9 or month==10 or month==11:
    return 2
  else:
    return 3

data['season']=data['month'].apply(lambda x: convert_month(x))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data['type_encoded']=le.fit_transform(data['type'])

data_ohe=pd.get_dummies(data=data, columns=['region'])

X=data_ohe.drop(['Date', 'AveragePrice', '4046', '4225', '4770',
        'Small Bags', 'Large Bags', 'XLarge Bags', 'type'], axis=1)
y=data_ohe['AveragePrice']


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)


pipe_RF=Pipeline([('scaler', StandardScaler()), 
                  ('rf', RandomForestRegressor())])

pipe_RF.fit(X_train, y_train)

y_pred_RF=pipe_RF.predict(X_test)

# Reads test data
organic_data = pd.read_csv('organic_test.csv')
organic_test=organic_data.set_index(pd.DatetimeIndex(organic_data['Date'])).drop('Date', axis=1)
# Reads test data
conventional_data = pd.read_csv('organic_test.csv')
conventional_test=conventional_data.set_index(pd.DatetimeIndex(conventional_data['Date'])).drop('Date', axis=1)


# Reads in saved arima model
organic_model = pickle.load(open('arima_model_organic.pkl', 'rb'))
conventional_model = pickle.load(open('arima_model_conventional.pkl', 'rb'))

# Reads in saved prophet model
df_pf_or = pd.read_csv('df_pf_or.csv', parse_dates=['ds'])
df_pf_or=df_pf_or.drop('Unnamed: 0', axis=1)
df_pf_or.y = df_pf_or.y.astype(float)

pf_model_or = Prophet(yearly_seasonality=True, \
            daily_seasonality=False, weekly_seasonality=False)

pf_model_organic=pf_model_or.fit(df_pf_or)

# Reads in saved prophet model
df_pf_con = pd.read_csv('df_pf_con.csv', parse_dates=['ds'])
df_pf_con=df_pf_con.drop('Unnamed: 0', axis=1)
df_pf_con.y = df_pf_con.y.astype(float)

pf_model_con = Prophet(yearly_seasonality=True, \
            daily_seasonality=False, weekly_seasonality=False)

pf_model_conventional=pf_model_con.fit(df_pf_con)

#=====================================================================================================================================================

if (selected == 'Introduction'):

    st.write("""
    # :books: Business Objective
    Hass, a company based in Mexico, specializes in producing a variety of avocados for selling in U.S.A. 
    They have been very successful over recent years and want to expand their business. 
    Thus they want build a reasonable model to predict the average price of avocado “Hass” in the U.S.A to consider the expansion of existing avocado farms in other regions.

    There are two types of avocados (conventional and organic) in the dataset and several different regions. 
    This allows us to do analysis for either conventional or organic avocados in different regions and/or the entire United States.

    There are 2 different approaches to solve this business objective:
    First approach: create a regression model using supervised machine learning algorithms such as Linear Regression, Random Forest, XGB Regressor so on to predict average price of avocado in the USA.
    Second approach: build a predictive model based on supervised time-series machine learning algorithms like Arima, Prophet, HoltWinters to predict average price of a particular avocado (organic or conventional) over time for a specific region in the USA.
    """)

    from PIL import Image 
    img = Image.open("images/avocado_1.jpg")
    st.image(img,width=700,caption='Streamlit Images')

    st.write("""
    # :chart_with_upwards_trend: Avocado price Forecasting
    This app used avocado.csv dataset as train and test data. 
    """)
    st.markdown('''
    This is a dashboard showing the *average prices* of different types of :avocado:  
    Data source: [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020)
    ''')

    st.info(" **OBJECTIVE:** To predict / forecast the average price of Avocado based on time series data using ***regression model***, ***ARIMA model***, ***PROPHET model*** ")


    st.header(':bar_chart: Summary statistics')

    st.dataframe(avocado_stats)

    st.header(':triangular_flag_on_post: Avocado Price and Total volume by geographies')

    b1, b2, b3, b4 = st.columns(4)
    region=b1.selectbox("Region", np.sort(data.region.unique()))
    avocado_type=b2.selectbox("Avocado Type", data.type.unique())

    start_date=b3.date_input("Start Date",
                                    data.Date.min().date(),
                                    min_value=data.Date.min().date(),
                                    max_value=data.Date.max().date(),
                                    )

    end_date=b4.date_input("End Date",
                                    data.Date.max().date(),
                                    min_value=data.Date.min().date(),
                                    max_value=data.Date.max().date(),
                                    )

    mask = ((data.region == region) &
            (data.type == avocado_type) &
            (data.Date >= pd.Timestamp(start_date)) &
            (data.Date <= pd.Timestamp(end_date)))
    filtered_data = data.loc[mask, :] 



    with st.form('line_chart'):
        filtered_avocado = filtered_data
        submitted = st.form_submit_button('Submit')
        check_box=st.checkbox("Included Sale Volume")

        if submitted:
            
            price_fig = px.line(filtered_avocado,
                            x='Date', y='AveragePrice',
                            color='type',
                            title=f'Avocado Prices in {data.type[0]}')
            st.plotly_chart(price_fig)
            
            # Show sale volume
            if check_box:	 
                volume_fig = px.line(filtered_avocado,
                                x='Date', y='Total Volume',
                                color='type',
                                title=f'Avocado Sale Volume in {data.type[0]}')           
                st.plotly_chart(volume_fig)


#=====================================================================================================================================================

elif (selected == 'Data Exploration'):
    st.header(':pushpin: Data Exploration')
    st.subheader(':memo: Avocado data summary')
    st.dataframe(data.describe())

    st.subheader(':chart_with_downwards_trend: Seasonality analysis')
    Byweekly = st.checkbox('Weekly')
    if Byweekly:
        st.success('🌎 Seasonality by weekly')
        from PIL import Image 
        img_weekly = Image.open("images/weekly.jpg")
        st.image(img_weekly,width=700,caption='Streamlit Images')
    Bymonthly = st.checkbox('Monthly')
    if Bymonthly:
        st.success('🌎 Seasonality by monthly')
        from PIL import Image 
        img_monthly = Image.open("images/monthly.jpg")
        st.image(img_monthly,width=700,caption='Streamlit Images')
    BySeason = st.checkbox('BySeason')
    if BySeason:
        st.success('🌎 Seasonality by monthly')
        from PIL import Image 
        img_seasonly = Image.open("images/seasonly.jpg")
        st.image(img_seasonly,width=700,caption='Streamlit Images')
    Byyearly = st.checkbox('Yearly')
    if Byyearly:
        st.success('🌎 Seasonality by yearly')
        from PIL import Image 
        img_yearly = Image.open("images/yearly.jpg")
        st.image(img_yearly,width=700,caption='Streamlit Images')
    
    st.subheader(':chart: Price analysis')
    columns = st.multiselect(label='Please select type of avocado to check the average price change by region', options=data.type.unique())
    if st.button("Generate Plot"):

        if columns==['organic']:
            
            st.write("classify by region, filter type=='organic' ")
            sns.set(style='whitegrid')
            fig1=plt.figure(figsize=(25,5))
            sns.boxplot(data=data[data['type']=='organic'], x='region', y='AveragePrice')
            plt.xticks(rotation=90)
            st.pyplot(fig1)

        elif columns==['conventional']:

            st.write("classify by region, filter type=='conventional' ")
            sns.set(style='whitegrid')
            fig2=plt.figure(figsize=(25,5))
            sns.boxplot(data=data[data['type']=='conventional'], x='region', y='AveragePrice')
            plt.xticks(rotation=90)
            st.pyplot(fig2)  

        elif columns==['organic', 'conventional']:
            
            st.write("classify by region, filter type==['conventinal','organic'] ")
            fig3=plt.figure(figsize=(25,5))
            sns.boxplot(data=data[data['type']=='organic'], x='region', y='AveragePrice')
            plt.xticks(rotation=90)
            st.pyplot(fig3)
            fig4=plt.figure(figsize=(25,5))
            sns.boxplot(data=data[data['type']=='conventional'], x='region', y='AveragePrice')
            plt.xticks(rotation=90)
            st.pyplot(fig4)
        else:
            st.write('please select again!')


    st.markdown('''
        grouped region from multiple states: show high sale volume

        - southest region
        - northest region
        - southcentral region
        - midsouth region
        - west region
                
        LosAngles city is belong to California state. California is the Largest avocado Consumer in the US
            
        ''')
    text=" In this exercise we will focus on [California] region"
    new_title = ':point_right:' + '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">' + text + '</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    



# https://www.webfx.com/tools/emoji-cheat-sheet/

#=====================================================================================================================================================    

elif (selected == 'Regression model'):
    st.header(':sparkles: Regression Model')
    st.markdown(':heart: Multiple regression models were run by Lazypredict to foind out which regressor is the best for this study')

    st.success('🌎 RandomForest is one of the the best regressor to be focus more this the following prediction')
    from PIL import Image 
    img3 = Image.open("images/lazypredict.jpg")
    st.image(img3,width=700,caption='Streamlit Images')

    # # load the model from disk
    # with open('randomforest_model.sav', 'rb') as pkl:
    #     rf_model = pickle.load(pkl)

    # y_pred_rf = rf_model.predict(X_test)

    # mae_rf = mean_absolute_error(y_test, y_pred_rf)

    # st.write(mae_rf)

    r2_score_RF=r2_score(y_test, y_pred_RF)
    # st.write(r2_score_RF)

    rae_RF=mean_absolute_error(y_test, y_pred_RF)
    # st.write(rae_RF)

    lst=[['Random Forest', r2_score_RF, rae_RF]]

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">The table below shows the result of RandomForest model:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    result_table = pd.DataFrame(lst, columns =['Model name', 'R2 score', 'Mean absolute error'])

    st.dataframe(result_table)

    st.info(':bookmark: Different between prediction and actual data')
    fig5=plt.figure(figsize=(25,15))
    plt.scatter(x=y_test, y=y_pred_RF)
    st.pyplot(fig5)

    

#=====================================================================================================================================================    

elif (selected == 'Arima model'):    
    st.header(':wind_chime: Arima Model')

    st.success('🌎 seasonal decomposition for organic type')
    from PIL import Image 
    arima_img1 = Image.open("images/arima_seasonal_decompose_organic.jpg")
    st.image(arima_img1,width=700,caption='Streamlit Images')

    st.success('🌎 diagnostics plot for organic type')
    from PIL import Image 
    arima_img2 = Image.open("images/arima_plot_diagnostics_organic.jpg")
    st.image(arima_img2,width=700,caption='Streamlit Images')


    st.success('🌎 ARIMA model prediction for organic type')
    from PIL import Image 
    arima_img3 = Image.open("images/arima_prediction_organic.jpg")
    st.image(arima_img3,width=700,caption='Streamlit Images')

    st.success('🌎 seasonal decomposition for conventional type')
    from PIL import Image 
    arima_img4 = Image.open("images/arima_seasonal_decompose_conventional.jpg")
    st.image(arima_img4,width=700,caption='Streamlit Images')

    st.success('🌎 diagnostics plot for conventional type')
    from PIL import Image 
    arima_img5 = Image.open("images/arima_plot_diagnostics_conventional.jpg")
    st.image(arima_img5,width=700,caption='Streamlit Images')


    st.success('🌎 ARIMA model prediction for conventional type')
    from PIL import Image 
    arima_img6 = Image.open("images/arima_prediction_conventional.jpg")
    st.image(arima_img6,width=700,caption='Streamlit Images')

    mae_or = 0.0810784207188943
    mape_or = 0.04596885301082209
    rmse_or = 0.09000986858549902

    mae_con = 0.134824331311639
    mape_con = 0.10298540343308708
    rmse_con = 0.1744429644006418

    lst=[['Arima_organic', mae_or, mape_or, rmse_or], ['Arima_conventional', mae_con, mape_con, rmse_con]]

    arima_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">The table below shows the result of ARIMA model:</p>'
    st.markdown(arima_title, unsafe_allow_html=True)

    arima_table = pd.DataFrame(lst, columns =['Model name', 'mae', 'mape', 'rmse'])

    st.dataframe(arima_table)


    # number of month to predict
    st.header(':flags: Arima Model Forecasting')
    avocado_type = st.radio("Select avocado type to predict:", ('organic', 'conventional'))
    no_months = st.slider('Number of months to predict', 1, 48, 12)


    def run():

        # number of month to predict
        #no_months = st.sidebar.slider('Number of months to predict', 1, 48, 12)

        if st.button('Run'):
            if (avocado_type == 'organic'):
                # Apply model to make predictions
                prediction = organic_model.predict(n_periods=len(organic_test)+no_months)
                st.success(f'Price prediction for the next {no_months} months is sucessful')
                st.dataframe(prediction)
                last_date=prediction.last_valid_index()
                last_price=prediction.iloc[-1]
                st.success(f'Organic Avocado price at {last_date} is {last_price:.2f} USD')
                
                last_prediction=prediction.tail(1)
                
                # fig = plt.figure()
                fig6=plt.figure(figsize=(15, 6))
                ax = fig6.add_subplot(1, 1, 1)
                ax.plot(prediction, label='Prediction price')
                ax.plot(last_prediction, 'ro')

                # Annotation
                ax.annotate(f'Organic Avocado price at {last_date} is {last_price:.2f} USD', (last_date, last_price), xytext=(0.8, 1.9),
                textcoords='offset points', arrowprops=dict(facecolor='green', shrink=0.5), ha='center')
                ax.legend()
                st.pyplot(fig6)
            else:
                # Apply model to make predictions
                prediction = conventional_model.predict(n_periods=len(conventional_test)+no_months)
                st.success(f'Price prediction for {no_months} months is sucessful')
                st.dataframe(prediction)
                last_date=prediction.last_valid_index()
                last_price=prediction.iloc[-1]
                st.success(f'Conventional Avocado price at {last_date} is {last_price:.2f} USD')
                
                last_prediction=prediction.tail(1)

                # fig = plt.figure()
                fig7=plt.figure(figsize=(15, 6))
                ax = fig7.add_subplot(1, 1, 1)
                ax.plot(prediction, label='Prediction price')
                ax.plot(last_prediction, 'ro')
                
                # Annotation
                ax.annotate(f'Organic Avocado price at {last_date} is {last_price:.2f} USD', (last_date, last_price), xytext=(0.8, 1.9),
                textcoords='offset points', arrowprops=dict(facecolor='green', shrink=0.5), ha='center')
                ax.legend()
                st.pyplot(fig7)
        else:
            st.error('Please click "Run" to predict the future price!')
    run()





#=====================================================================================================================================================    

elif (selected == 'Prophet model'): 
    st.header(':wind_chime: Prophet Model') 
 
    st.success('🌎 Prophet components plot for organic type')
    from PIL import Image 
    prophet_img1 = Image.open("images/prophet_plot_components_organic.jpg")
    st.image(prophet_img1,width=700,caption='Streamlit Images')

    st.success('🌎 Prophet prediction plot for organic type')
    from PIL import Image 
    prophet_img2 = Image.open("images/prophet_prediction_organic.jpg")
    st.image(prophet_img2,width=700,caption='Streamlit Images')


    st.success('🌎 Prophet components plot for conventional type')
    from PIL import Image 
    prophet_img3 = Image.open("images/prophet_plot_components_conventional.jpg")
    st.image(prophet_img3,width=700,caption='Streamlit Images')

    st.success('🌎 Prophet prediction plot for conventional type')
    from PIL import Image 
    prophet_img4 = Image.open("images/prophet_prediction_conventional.jpg")
    st.image(prophet_img4,width=700,caption='Streamlit Images')

    mae_organic = 0.20912640616673214
    mape_organic = 0.12323229223830527
    rmse_organic = 0.23976214502612603

    mae_conventional = 0.23568233062260677
    mape_conventional = 0.18630272896575167
    rmse_conventional = 0.2967862427328833

    lst1=[['Prophet_Organic', mae_organic, mape_organic, rmse_organic], ['Prophet_Conventional', mae_conventional, mape_conventional, rmse_conventional]]

    prophet_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">The table below shows the result of PROPHET model:</p>'
    st.markdown(prophet_title, unsafe_allow_html=True)

    prophet_table = pd.DataFrame(lst1, columns =['Model name', 'mae', 'mape', 'rmse'])

    st.dataframe(prophet_table)


    # number of month to predict
    st.header(':flags: Prophet Model Forecasting')
    avocado_type = st.radio("Select avocado type to predict:", ('organic', 'conventional'))
    no_months_2 = st.slider('Number of months to predict', 1, 48, 12)

    # xx months in the time frame of test dataset
    import datetime 
    import dateutil.relativedelta
    from datetime import date
    lastmonth=datetime.date(2018, 3, 1)
    nextmonth=lastmonth + dateutil.relativedelta.relativedelta(months=+no_months_2)

    new_months = pd.date_range(lastmonth,nextmonth, 
                freq='M').strftime("%Y-%m-%d").tolist()    
    
    forecast = pd.DataFrame(new_months)
    forecast.columns = ['ds']
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    


    def run():

        # number of month to predict
        #no_months = st.sidebar.slider('Number of months to predict', 1, 48, 12)

        if st.button('Run'):
            if (avocado_type == 'organic'):
                # Apply model to make predictions
                future_forecast_organic = pf_model_organic.predict(forecast)
                future_forecast_organic=future_forecast_organic[['ds','yhat']]

                st.success(f'Price prediction for the next {no_months_2} months is sucessful')
                st.dataframe(future_forecast_organic)
                last_date=future_forecast_organic.ds.iloc[-1]
                last_price=future_forecast_organic.yhat.iloc[-1]
                st.success(f'Organic Avocado price at {last_date} is {last_price:.2f} USD')

                # fig = plt.figure()
                future_forecast_organic_dt=future_forecast_organic[['ds','yhat']].copy()
                future_forecast_organic_dt=future_forecast_organic_dt.set_index(pd.DatetimeIndex(future_forecast_organic_dt['ds']))
                future_forecast_organic_dt=future_forecast_organic_dt.drop(['ds'], axis=1)
                last_prediction=future_forecast_organic_dt.tail(1)

               

                prophet_fig5=plt.figure(figsize=(15, 6))
                ax = prophet_fig5.add_subplot(1, 1, 1)
                ax.plot(future_forecast_organic_dt.yhat, label='Prediction price')
                ax.plot(last_prediction, 'ro')
                # Annotation
                ax.annotate(f'Organic Avocado price at {last_date} is {last_price:.2f} USD', (last_date, last_price), xytext=(0.8, 1.9),
                textcoords='offset points', arrowprops=dict(facecolor='green', shrink=0.5), ha='center')
                ax.legend()
                st.pyplot(prophet_fig5)
            else:
                # Apply model to make predictions
                future_forecast_conventional = pf_model_conventional.predict(forecast)
                future_forecast_conventional=future_forecast_conventional[['ds','yhat']]
                
                st.success(f'Price prediction for the next {no_months_2} months is sucessful')
                st.dataframe(future_forecast_conventional)
                last_date=future_forecast_conventional.ds.iloc[-1]
                last_price=future_forecast_conventional.yhat.iloc[-1]
                st.success(f'Conventional Avocado price at {last_date} is {last_price:.2f} USD')

                # fig = plt.figure()
                future_forecast_conventional_dt=future_forecast_conventional[['ds','yhat']].copy()
                future_forecast_conventional_dt=future_forecast_conventional_dt.set_index(pd.DatetimeIndex(future_forecast_conventional_dt['ds']))
                future_forecast_conventional_dt=future_forecast_conventional_dt.drop(['ds'], axis=1)
                last_prediction=future_forecast_conventional_dt.tail(1)

                prophet_fig6=plt.figure(figsize=(15, 6))
                ax = prophet_fig6.add_subplot(1, 1, 1)
                ax.plot(future_forecast_conventional_dt.yhat, label='Prediction price')
                ax.plot(last_prediction, 'ro')
                # Annotation
                ax.annotate(f'Organic Avocado price at {last_date} is {last_price:.2f} USD', (last_date, last_price), xytext=(0.8, 1.9),
                textcoords='offset points', arrowprops=dict(facecolor='green', shrink=0.5), ha='center')
                ax.legend()
                st.pyplot(prophet_fig6)
        else:
            st.error('Please click "Run" to predict the future price!')
    run()
    
#=====================================================================================================================================================    

elif (selected == 'Evaluation'): 
    st.snow()
    st.header('Evaluation Model') 

#=====================================================================================================================================================    

elif (selected == 'Thank You'): 
    st.header('THANK YOU FOR YOUR LISTENING') 

    st.balloons()