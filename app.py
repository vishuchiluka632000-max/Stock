import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, feedparser
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

st.set_page_config("Stock Analyzer Pro", layout="wide")

# ---------------- STYLE ----------------

st.markdown("""
<style>
body {background:#0e1117;color:white}
.card {background:#161b22;padding:16px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")

# ---------------- SAFE SEARCH ----------------

popular = {
    "Reliance Industries":"RELIANCE.NS",
    "TCS":"TCS.NS",
    "Infosys":"INFY.NS",
    "HDFC Bank":"HDFCBANK.NS",
    "ICICI Bank":"ICICIBANK.NS",
    "ITC":"ITC.NS",
    "State Bank of India":"SBIN.NS",
    "Bharti Airtel":"BHARTIARTL.NS",
    "Larsen & Toubro":"LT.NS",
    "Indian Energy Exchange":"IEX.NS"
}

search = st.text_input("Search company")

matches = [k for k in popular if search.lower() in k.lower()]

if not matches:
    st.stop()

company = st.selectbox("Select Company", matches)
symbol = popular[company]

# ---------------- FETCH ----------------

stock = yf.Ticker(symbol)
df = stock.history(period="max")

if df.empty:
    st.error("No market data found")
    st.stop()

info = stock.info
df.reset_index(inplace=True)

# ---------------- TIME RANGE ----------------

ranges = {
    "1D":1,"1W":5,"10D":10,"1M":22,"3M":66,"6M":132,
    "1Y":252,"3Y":756,"5Y":1260,"ALL":len(df)
}

choice = st.radio("Range", list(ranges.keys()), horizontal=True)
plot_df = df.tail(ranges[choice])

# ---------------- CANDLE ----------------

fig = go.Figure(go.Candlestick(
    x=plot_df["Date"],
    open=plot_df["Open"],
    high=plot_df["High"],
    low=plot_df["Low"],
    close=plot_df["Close"]
))

fig.update_layout(template="plotly_dark", height=520)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"### Market Cap: ₹{round(info.get('marketCap',0)/1e7,2)} Cr")

# ---------------- FINANCIALS ----------------

def to_cr(df):
    return df.applymap(lambda x: f"{x/1e7:.2f} Cr" if pd.notna(x) else "")

st.subheader("📑 Financial Statements")

t1,t2,t3 = st.tabs(["Income","Balance Sheet","Cash Flow"])

with t1: st.dataframe(to_cr(stock.financials))
with t2: st.dataframe(to_cr(stock.balance_sheet))
with t3: st.dataframe(to_cr(stock.cashflow))

# ---------------- RATIOS ----------------

metrics = {
    "PE Ratio":info.get("trailingPE"),
    "PB Ratio":info.get("priceToBook"),
    "ROE %":info.get("returnOnEquity",0)*100,
    "Profit Margin %":info.get("profitMargins",0)*100,
    "Debt to Equity":info.get("debtToEquity"),
    "Current Ratio":info.get("currentRatio"),
    "Revenue Growth %":info.get("revenueGrowth",0)*100
}

def status(v, good, ok):
    if v is None: return "N/A"
    if v>=good: return "Excellent"
    if v>=ok: return "Good"
    return "Bad"

table = [
    ["PE Ratio",metrics["PE Ratio"],status(metrics["PE Ratio"],25,15)],
    ["PB Ratio",metrics["PB Ratio"],status(metrics["PB Ratio"],5,2)],
    ["ROE %",metrics["ROE %"],status(metrics["ROE %"],18,12)],
    ["Profit Margin %",metrics["Profit Margin %"],status(metrics["Profit Margin %"],15,8)],
    ["Debt to Equity",metrics["Debt to Equity"],status(1-(metrics["Debt to Equity"] or 1),0.6,0.3)],
    ["Current Ratio",metrics["Current Ratio"],status(metrics["Current Ratio"],2,1)],
    ["Revenue Growth %",metrics["Revenue Growth %"],status(metrics["Revenue Growth %"],15,8)]
]

ratio_df = pd.DataFrame(table, columns=["Metric","Value","Health"])

st.subheader("📊 Financial Ratios")
st.table(ratio_df)

strong = (ratio_df["Health"]=="Excellent").sum()
st.success(f"{strong}/{len(ratio_df)} metrics strong — fundamentals look {'strong' if strong>=4 else 'average'}")

# ---------------- FORECAST ----------------

df["t"]=range(len(df))
X=df[["t"]]
y=df["Close"]

model=LinearRegression().fit(X,y)

future=np.arange(len(df),len(df)+180).reshape(-1,1)
pred=model.predict(future)

forecast=pd.DataFrame({"Day":range(180),"Price":pred})

st.subheader("📈 6 Month Projection")
st.line_chart(forecast.set_index("Day"))

# ---------------- NEWS ----------------

st.subheader("📰 Latest News")

def news(q):
    feed = feedparser.parse(
        f"https://news.google.com/rss/search?q={q}
