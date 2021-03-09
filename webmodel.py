# my first Streamlit App
import os

# ml libs
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

# visualization libraries
# from matplotlib import pyplot as plt
import pickle

# streamlit
import streamlit as st
import altair

# data access
import pandas as pd
import numpy as np

# fb prophet
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


def get_home_data(zip_code):
    median=pd.read_csv('medianSalesPrice.csv', dtype={"Bedrooms": int, "RegionName": "string"})
    mew_dimension=median[median['RegionName']==zip_code]
    mew_dimension=mew_dimension.drop(columns=['RegionID','SizeRank','Bedrooms','RegionType','StateName','State','Metro','CountyName'])
    mew_dimension=mew_dimension.mean()
    mew_dimension=pd.DataFrame(mew_dimension)
    mew_dimension.index=pd.to_datetime(mew_dimension.index)
    return mew_dimension


def stream_graph(df):
    st.line_chart(df)


def predict_model(df):
    model = Prophet()
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'ds',0:'y'}) 
    model.fit(df)
    forecast_1=model.make_future_dataframe(periods= 12, freq='M')
    forecast_1=model.predict(forecast_1)
    st.pyplot(model.plot_components(forecast_1))
    fig=model.plot(forecast_1)
    st.pyplot(fig)


###Page Format and Layout

#st.write('Miami-Dade Real Estate 2021 Forecast')

### init a title
st.title('Miami-Dade Real Estate Prices')
#st.header('Prophet')
st.subheader('The model was built using Facebook Prophet Method')

#miami=pd.read_csv('medianSalesPrice.csv', dtype={"Bedrooms": int, "RegionName": "string"})
#miami=miami[miami['CountyName']=='Miami-Dade County']
#st.table(miami.head())
#st.line_chart(miami)


selected_item=st.sidebar.selectbox('Chose Zip Code to run model', ("33160",
"33139",
"33186",
"33012",
"33157",
"33015",
"33141",
"33178",
"33033",
"33176",
"33125",
"33142",
"33180",
"33130",
"33179",
"33126",
"33165",
"33161",
"33175",
"33177",
"33134",
"33196",
"33032",
"33131",
"33018",
"33172",
"33016",
"33155",
"33133",
"33193",
"33014",
"33147",
"33135",
"33010",
"33140",
"33169",
"33137",
"33143",
"33162",
"33055",
"33156",
"33138",
"33183",
"33173",
"33056",
"33145",
"33030",
"33174",
"33054",
"33127",
"33150",
"33154",
"33144",
"33013",
"33132",
"33166",
"33189",
"33181",
"33129",
"33168",
"33034",
"33149",
"33136",
"33184",
"33167",
"33187",
"33146",
"33170",
"33182",
"33128",
"33158",
"33031",
"33160",
"33139",
"33186",
"33012",
"33157",
"33015",
"33141",
"33178",
"33033",
"33176",
"33125",
"33142",
"33180",
"33130",
"33179",
"33126",
"33165",
"33161",
"33175",
"33177",
"33134",
"33196",
"33032",
"33131",
"33018",
"33172",
"33016",
"33155",
"33133",
"33193",
"33014",
"33147",
"33135",
"33010",
"33140",
"33169",
"33137",
"33143",
"33162",
"33055",
"33156",
"33138"
))
st.write("Zip code: ",selected_item)
data_city=get_home_data(selected_item)

st.dataframe(data_city)

predict_model(data_city)

