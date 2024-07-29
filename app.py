from datetime import datetime, timedelta
import time
import threading

import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import streamlit.components.v1 as components

class App:

    def __init__(self):
        self.all_stocks_data = pd.read_csv("stock_tickers.csv")
        self.region = None
        self.selected_region_data = None
        self.selected_stock_name = None
        self.selected_stock_data = None
        self.data = None
        self.tab0 = None
        self.tab1 = None
        self.tab2 = None
        self.tab3 = None
        st.set_page_config(
            page_title="Stock Value Prediction",
            page_icon="📈"
        )

    def main(self):
        
        components.html("<h1 style='text-align: center; font-family: Arial; color:#000; -webkit-text-stroke-width: 1px; -webkit-text-stroke-color: #fff';>Stock Value Prediction</h1>", height = 100)
        
        font_css = """
        <style>
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
        }
        </style>
        """
        hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
        st.markdown(font_css, unsafe_allow_html=True)
        # st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

        countries = np.concatenate(([""], self.all_stocks_data["Country"].unique()))

        self.region = st.selectbox('Select Stock Region', countries)

        self.selected_region_data = self.all_stocks_data[self.all_stocks_data["Country"] == self.region]
        
        self.selected_stock_name = st.selectbox('Select Stock', self.selected_region_data["Symbol"] + " - " + self.selected_region_data["Name"].str.slice(stop=50))

        if self.selected_stock_name != None:
            self.selected_stock_data = self.selected_region_data[self.selected_region_data["Symbol"] == self.selected_stock_name.split(" ")[0]]

        if self.selected_stock_name == None:
            start_data = st.slider("Choose the range of past year data (in years)", 1, 10, 1, disabled=True)
        else:
            start_data = st.slider("Choose the range of past year data (in years)", 1, 10, 1, key="start")
        
        _, col, _ = st.columns([1, 1, 1])
        if self.selected_stock_name == None:
            col.button("Get Data", type="primary", use_container_width=True, key="btn1", disabled=True)  
        else:
            col.button("Get Data", on_click=self.downloadData, args = (start_data,), use_container_width=True, type="primary", key="btn1")

        st.divider()

        self.tab0, self.tab1, self.tab2, self.tab3 = st.tabs(["Information".center(15,"\u2001"), "Data".center(10,"\u2001"), "Moving Avg".center(15,"\u2001"), "Prediction".center(15,"\u2001")])

    
    def downloadData(self, start):

        self.getMetadata()

        start = (datetime.now() - timedelta(365*start)).strftime('%Y-%m-%d')
        end = (datetime.now()).strftime('%Y-%m-%d')

        self.data = yf.download(self.selected_stock_data.iloc[0,0], start, end)

        self.tab1.subheader(self.selected_stock_data.iloc[0,1] + ": Stock Data")
        self.tab1.dataframe(self.data, width=1000)

        time.sleep(4)
        self.getMovingAvg()
        
    def getMetadata(self):
        st.session_state["run"] = True
        data = {"Values":{
                    "Stock Symbol": self.selected_stock_data["Symbol"].iloc[0],
                    "Stock Name": self.selected_stock_data["Name"].iloc[0],
                    "Stock Region": self.selected_stock_data["Country"].iloc[0],
                    "IPO Year": int(self.selected_stock_data["IPO Year"].iloc[0]) if not np.isnan(self.selected_stock_data["IPO Year"].iloc[0]) else "-",
                    "Sector": self.selected_stock_data["Sector"].iloc[0] if not pd.isna(self.selected_stock_data["Sector"].iloc[0]) else "-",
                    "Industry": self.selected_stock_data["Industry"].iloc[0] if not pd.isna(self.selected_stock_data["Sector"].iloc[0]) else "-"
                    }
                }
        self.tab0.table(data)

    def getMovingAvg(self):
        if not self.data.empty:

            _lock = threading.Lock()
            with _lock:
                mov_avg_50 = self.data["Close"].rolling(50).mean()
                mov_avg_100 = self.data["Close"].rolling(100).mean()
                mov_avg_200 = self.data["Close"].rolling(200).mean()

                fig = plt.figure(figsize=(12,9), dpi=500)
                plt.plot(mov_avg_50, "r", label="50 day moving avg", alpha=0.8)
                plt.plot(mov_avg_100, "g", label="100 day moving avg", alpha=0.8)
                plt.plot(mov_avg_200, "b", label="200 day moving avg", alpha=0.8)
                plt.plot(self.data["Close"], "Black", label="Stock Closing Price", alpha=0.8)
                plt.title("Actual Price vs Moving Average")
                plt.xticks(rotation=45)
                plt.xlabel("Calendar Years")
                plt.ylabel("Stock Price in $")
                plt.legend()
                plt.grid()
                self.tab2.pyplot(fig)

            time.sleep(4)
            self.getPrediction()
        

    def getPrediction(self):

        tf.random.set_seed(0)
        y = self.data['Close'].fillna('ffill')
        y = y.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)

        # n_lookback = 90  # length of input sequences (lookback period)
        n_lookback = len(self.data)//10  # length of input sequences (lookback period)
        # n_forecast = 30  # length of output sequences (forecast period)
        n_forecast = len(self.data)//100  # length of output sequences (forecast period)

        X = []
        Y = []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        X = np.array(X)
        Y = np.array(Y)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(Dropout(0.5))
        model.add(LSTM(units=50))
        model.add(Dense(n_forecast))

        model.compile(loss='mean_squared_error', optimizer='adam')

        with self.tab3.container():
            with st.spinner('Model Training in Progress...'):
                history = model.fit(X, Y, epochs=10, batch_size=32, verbose=2)

        X_ = y[-n_lookback:]  # last available input sequence
        X_ = X_.reshape(1, n_lookback, 1)

        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)

        df_past = self.data[['Close']].reset_index()
        df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Forecast'] = np.nan
        df_past.iloc[2,-1] = df_past['Actual'].iloc[-1]

        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan

        results = pd.concat([df_past, df_future]).set_index("Date")

        _lock = threading.Lock()
        with _lock:
            fig = plt.figure(figsize=(16, 12), dpi=300)
            plt.plot(results.iloc[-(30+len(self.data)//100)])
            # plt.xlim(datetime.now() - timedelta(len(self.data)//100),datetime.now() + timedelta(len(self.data)//100))
            # plt.xlim(datetime.now() - timedelta(30),datetime.now() + timedelta(len(self.data)//100))
            plt.grid()
            plt.xticks(rotation=45)
            plt.xlabel("Calendar Days")
            plt.ylabel("Stock Price in $")
            plt.legend(results.columns)
            
            self.tab3.pyplot(fig) 

if __name__ == "__main__":
    app = App()
    app.main()
    
