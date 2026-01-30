import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import feedparser
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config("Stock Analyzer Pro", layout="wide")

st.markdown("""
<style>
.stApp{
background-image:url("https://images.unsplash.com/photo-1569025690938-a00729c9e1b3");
background-size:cover;
background-attachment:fixed;
}
.block-container{background:#0e1117;padding:2rem;border-radius:15px}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")

# ---------------- STOCK LIST ----------------
stocks = {
    "Reliance Industries":"RELIANCE.NS",
    "Tata Motors":"TATAMOTORS.NS",
    "TCS":"TCS.NS",
    "Infosys":"INFY.NS",
    "HDFC Bank":"HDFCBANK.NS",
    "ICICI Bank":"ICICIBANK.NS",
    "ITC":"ITC.NS",
    "State Bank of India":"SBIN.NS",
    "Bharti Airtel":"BHARTIARTL.NS",
    "Larsen & Toubro":"LT.NS",
    "Asian Paints":"ASIANPAINT.NS",
    "HUL":"HINDUNILVR.NS"
}

names = list(stocks.keys())

# ---------------- SMART SEARCH (NO LIBRARY) ----------------
query = st.text_input("Search stock name")

matches = [n for n in names if query.lower() in n.lower()]

if not matches:
    st.stop()

company = st.selectbox("Select company", matches)
symbol = stocks[company]

# ---------------- TIMEFRAME ----------------
periods = {
    "1D":"1d","1W":"5d","1M":"1mo","3M":"3mo",
    "6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y","ALL":"max"
}

tf = st.radio("Timeframe", list(periods.keys()), horizontal=True)

data = yf.download(symbol, period=periods[tf], progress=False)

# ---------------- CHART SWITCH ----------------
chart_type = st.radio("Chart Type", ["Candlestick","Line"], horizontal=True)

fig = go.Figure()

if chart_type=="Candlestick":
    fig.add_candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )
else:
    fig.add_scatter(x=data.index,y=data["Close"])

fig.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig,use_container_width=True)

# ---------------- MARKET CAP ----------------
info = yf.Ticker(symbol).info
mcap = info.get("marketCap",0)/1e7
st.metric("Market Cap (₹ Cr)", f"{mcap:,.0f}")

# ---------------- FINANCIALS ----------------
st.subheader("📑 Financial Statements (₹ Cr)")

ticker = yf.Ticker(symbol)

def cr(df):
    if df is None: return pd.DataFrame()
    return (df/1e7).round(1)

t1,t2,t3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with t1: st.dataframe(cr(ticker.financials))
with t2: st.dataframe(cr(ticker.balance_sheet))
with t3: st.dataframe(cr(ticker.cashflow))

# ---------------- RATIOS ----------------
def v(x): return round(x,2) if x else 0

ratios = {
    "PE Ratio":v(info.get("trailingPE")),
    "ROE %":v(info.get("returnOnEquity",0)*100),
    "Profit Margin %":v(info.get("profitMargins",0)*100),
    "Debt to Equity":v(info.get("debtToEquity")),
    "Current Ratio":v(info.get("currentRatio")),
    "Revenue Growth %":v(info.get("revenueGrowth",0)*100)
}

def status(k,v):
    if "PE" in k: return "Excellent" if v<20 else "Good" if v<30 else "Bad"
    if "ROE" in k: return "Excellent" if v>18 else "Good" if v>12 else "Bad"
    if "Margin" in k: return "Excellent" if v>15 else "Good" if v>8 else "Bad"
    if "Debt" in k: return "Excellent" if v<0.5 else "Good" if v<1 else "Bad"
    return "Good"

df_ratios = pd.DataFrame(
    [[k,v,status(k,v)] for k,v in ratios.items()],
    columns=["Ratio","Value","Health"]
)

st.subheader("📊 Financial Ratios")
st.dataframe(df_ratios)

score = (df_ratios["Health"]=="Excellent").sum()
st.info("Overall company looks " + ("Strong" if score>=3 else "Moderate"))

# ---------------- FORECAST ----------------
st.subheader("📈 Future Price Projection")

df = data.reset_index()
df["t"]=np.arange(len(df))

model = LinearRegression().fit(df[["t"]], df["Close"])

days = st.slider("Projection days",30,180,90)
future = np.arange(len(df),len(df)+days).reshape(-1,1)
pred = model.predict(future)

forecast = pd.DataFrame({"Day":range(days),"Price":pred})
st.line_chart(forecast.set_index("Day"))

# ---------------- NEWS ----------------
st.subheader("📰 Latest Stock News")

def news(q):
    url = f"https://news.google.com/rss/search?q={q}+stock"
    feed = feedparser.parse(url)
    for e in feed.entries[:6]:
        dt = datetime(*e.published_parsed[:6])
        st.markdown(f"**{e.title}**  \n🕒 {dt.strftime('%d %b %Y %H:%M')}  \n[Read]({e.link})")

news(company)
