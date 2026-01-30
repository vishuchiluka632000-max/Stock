import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import ta

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Smart Stock Analyzer")

symbol = st.text_input("Enter Stock Symbol (ex: TCS.NS, RELIANCE.NS)", "TCS.NS")

try:
    stock = yf.Ticker(symbol)
    df = stock.history(period="5y")

    if df.empty:
        st.error("No data found.")
        st.stop()

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    fig = px.line(df, y="Close", title="Price Chart")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Moving Averages")
    st.line_chart(df[["Close","MA50","MA200"]])

    X = np.arange(len(df)).reshape(-1,1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X,y)

    future = np.arange(len(df), len(df)+180).reshape(-1,1)
    preds = model.predict(future)

    forecast_df = pd.DataFrame({"Prediction":preds})
    st.subheader("6 Month Forecast")
    st.line_chart(forecast_df)

except Exception as e:
    st.error("Error fetching data")
