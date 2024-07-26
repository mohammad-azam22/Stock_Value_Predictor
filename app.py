from datetime import datetime, timedelta
import time

import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.saving import load_model
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import streamlit.components.v1 as components

import threading

class App:

    def __init__(self):
        self.all_stocks_data = pd.read_csv("A:\myvenv\myfiles\stock_tickers.csv")
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

        components.html("<h1 style='text-align: center; font-family: Arial; color:#000;'>Stock Value Prediction</h1>", height = 100)
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
        # st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
        st.write(font_css, unsafe_allow_html=True)

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
            _lock = threading.Lock()
            with _lock:
                mov_avg_50 = self.data["Close"].rolling(50).mean()
                mov_avg_100 = self.data["Close"].rolling(100).mean()
                mov_avg_200 = self.data["Close"].rolling(200).mean()

                fig = plt.figure(figsize=(12,9), dpi=500)
                plt.plot(mov_avg_50, "r", label="50 day moving avg")
                plt.plot(mov_avg_100, "g", label="100 day moving avg")
                plt.plot(mov_avg_200, "b", label="200 day moving avg")
                plt.plot(self.data["Close"], "Black", label="Stock Closing Price")
                plt.title("Actual Price vs 50 day Moving Avg. 100 day Moving Avg. vs 200 day Moving Avg.")
                plt.xlabel("Calendar Years")
                plt.ylabel("Stock Price in USD")
                plt.legend()
                plt.grid()
                self.tab2.pyplot(fig)
            
            time.sleep(4)
            self.getPrediction()

    def getPrediction(self):
        
        self.data.dropna(inplace=True)
        train_data = pd.DataFrame(self.data["Close"][0 : int(len(self.data)*0.80)])    # getting initial 80% data as the training data
        test_data = pd.DataFrame(self.data["Close"][int(len(self.data)*0.80) : ])    # getting remaining 20% data as the testing data

        scaler = MinMaxScaler(feature_range=(0,1))

        past_100_days = train_data.tail(100)
        test_data = pd.concat([past_100_days, test_data], ignore_index=True)

        test_data_scaled = scaler.fit_transform(test_data)
        scale = 1/scaler.scale_

        x = []
        y = []
        for i in range(100, test_data_scaled.shape[0]):
            x.append(test_data_scaled[i-100:i])
            y.append(test_data_scaled[i,0])

        x = np.array(x)
        y = np.array(y)

        model = load_model("A:\myvenv\myfiles\Stock_Prediction_Model.keras")
        predict = model.predict(x)

        predict = predict * scale
        y = y * scale

        _lock = threading.Lock()
        with _lock:
            self.tab3.subheader("Actual Stock Price vs Predicted Stock Price")
            fig4 = plt.figure(figsize=(8,6))
            plt.plot(predict, "r", label="Predicted Stock Price")
            plt.plot(y, "g", label="Actual Stock Price")
            plt.xlabel("Weeks")
            plt.ylabel("Stock Price in USD")
            plt.legend()
            plt.grid()
            self.tab3.pyplot(fig4)

    
if __name__ == "__main__":
    app = App()
    app.main()