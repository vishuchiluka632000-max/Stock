import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Analyzer Pro", layout="wide")
st.title("📊 Stock Analyzer Pro (Screener Style)")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    stock1 = st.text_input("Stock 1 (ex: RELIANCE.NS)", "RELIANCE.NS")
with col2:
    stock2 = st.text_input("Stock 2 (optional)", "")

# ---------------- DATA FETCH ----------------
def load_stock(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="5y")
    info = stock.info
    return df, info

try:
    df1, info1 = load_stock(stock1)
    if df1.empty:
        st.error("Invalid stock symbol")
        st.stop()
except:
    st.error("Error fetching stock data")
    st.stop()

# ---------------- TECHNICALS ----------------
df1['MA50'] = df1['Close'].rolling(50).mean()
df1['MA200'] = df1['Close'].rolling(200).mean()

# ---------------- PERFORMANCE ----------------
def CAGR(df):
    start = df['Close'].iloc[0]
    end = df['Close'].iloc[-1]
    years = len(df)/252
    return ((end/start)**(1/years)-1)*100

# ---------------- RATIOS ----------------
def safe(val):
    return round(val,2) if val else 0

ratios = {
    "Market Cap": info1.get("marketCap"),
    "PE Ratio": safe(info1.get("trailingPE")),
    "ROE %": safe(info1.get("returnOnEquity",0)*100),
    "Profit Margin %": safe(info1.get("profitMargins",0)*100),
    "Debt to Equity": safe(info1.get("debtToEquity")),
}

# ---------------- CHART ----------------
price_chart = px.line(df1, y="Close", title=f"{stock1} Price")
ma_chart = px.line(df1, y=["Close","MA50","MA200"], title="Moving Averages")

st.plotly_chart(price_chart, use_container_width=True)
st.plotly_chart(ma_chart, use_container_width=True)

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("CAGR", f"{CAGR(df1):.2f}%")
col2.metric("Current Price", f"₹{df1['Close'].iloc[-1]:.2f}")
col3.metric("52W High", f"₹{df1['Close'].tail(252).max():.2f}")

# ---------------- RATIOS TABLE ----------------
st.subheader("📈 Key Financial Ratios")
st.table(pd.DataFrame(ratios.items(), columns=["Metric","Value"]))

# ---------------- FORECAST ----------------
df_reset = df1.reset_index()
df_reset['t'] = np.arange(len(df_reset))
X = df_reset[['t']]
y = df_reset['Close']

model = LinearRegression()
model.fit(X,y)

future = np.arange(len(df_reset), len(df_reset)+180).reshape(-1,1)
preds = model.predict(future)

forecast_df = pd.DataFrame({"Day": range(180), "Predicted Price": preds})
forecast_chart = px.line(forecast_df, x="Day", y="Predicted Price", title="6 Month Forecast")

st.plotly_chart(forecast_chart, use_container_width=True)

# ---------------- HEALTH SCORE ----------------
score = 0
if df1['MA50'].iloc[-1] > df1['MA200'].iloc[-1]: score += 40
if CAGR(df1) > 12: score += 30
if ratios['PE Ratio'] and ratios['PE Ratio'] < 25: score += 30

st.subheader("📌 Stock Health Score")
st.progress(score/100)
st.write(f"Score: {score}/100")

# ---------------- SUMMARY ----------------
if score >= 70:
    st.success("Strong business + positive trend. Looks fundamentally healthy.")
elif score >= 40:
    st.warning("Average performance. Needs monitoring.")
else:
    st.error("Weak trend or expensive valuation.")

# ---------------- COMPARISON ----------------
if stock2:
    try:
        df2, info2 = load_stock(stock2)
        comp = pd.DataFrame({
            stock1: [CAGR(df1), df1['Close'].iloc[-1]],
            stock2: [CAGR(df2), df2['Close'].iloc[-1]]
        }, index=["CAGR %","Current Price"])

        st.subheader("⚔ Stock Comparison")
        st.table(comp)
    except:
        st.warning("Could not fetch second stock")
