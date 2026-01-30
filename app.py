import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from rapidfuzz import process
import feedparser
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config("Stock Analyzer Pro", layout="wide")

BACKGROUND = """
<style>
.stApp {
background-image:url("https://images.unsplash.com/photo-1569025690938-a00729c9e1b3");
background-size:cover;
background-attachment:fixed;
}
.block-container{background:#0e1117;padding:2rem;border-radius:15px}
</style>
"""
st.markdown(BACKGROUND, unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")

# ---------------- STOCK UNIVERSE ----------------
stocks = {
    "Reliance Industries":"RELIANCE.NS",
    "Tata Motors":"TATAMOTORS.NS",
    "TCS":"TCS.NS",
    "Infosys":"INFY.NS",
    "HDFC Bank":"HDFCBANK.NS",
    "ICICI Bank":"ICICIBANK.NS",
    "ITC":"ITC.NS",
    "SBI":"SBIN.NS",
    "Bharti Airtel":"BHARTIARTL.NS",
    "Larsen & Toubro":"LT.NS",
    "Asian Paints":"ASIANPAINT.NS",
    "HUL":"HINDUNILVR.NS"
}

# ---------------- SMART SEARCH ----------------
company_list = list(stocks.keys())
query = st.text_input("Search stock name")

suggestions = []
if query:
    suggestions = [x[0] for x in process.extract(query, company_list, limit=6)]

if not suggestions:
    st.stop()

company = st.selectbox("Select company", suggestions)
symbol = stocks[company]

# ---------------- TIMEFRAME ----------------
period_map = {
    "1D":"1d","1W":"5d","1M":"1mo","3M":"3mo",
    "6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y","ALL":"max"
}

tf = st.radio("Timeframe", list(period_map.keys()), horizontal=True)

data = yf.download(symbol, period=period_map[tf], progress=False)

# ---------------- CANDLE / LINE SWITCH ----------------
view = st.radio("Chart Type", ["Candlestick","Line"], horizontal=True)

fig = go.Figure()

if view=="Candlestick":
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

def clean(df):
    if df is None: return pd.DataFrame()
    return (df/1e7).round(1)

tab1,tab2,tab3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with tab1: st.dataframe(clean(ticker.financials))
with tab2: st.dataframe(clean(ticker.balance_sheet))
with tab3: st.dataframe(clean(ticker.cashflow))

# ---------------- RATIOS ----------------
def val(x): return round(x,2) if x else 0

ratios = {
    "PE":val(info.get("trailingPE")),
    "ROE %":val(info.get("returnOnEquity",0)*100),
    "Profit Margin %":val(info.get("profitMargins",0)*100),
    "Debt/Equity":val(info.get("debtToEquity")),
    "Current Ratio":val(info.get("currentRatio")),
    "Revenue Growth %":val(info.get("revenueGrowth",0)*100)
}

def grade(name,v):
    if name=="PE":
        return "Excellent" if v<20 else "Good" if v<30 else "Bad"
    if "ROE" in name:
        return "Excellent" if v>18 else "Good" if v>12 else "Bad"
    if "Margin" in name:
        return "Excellent" if v>15 else "Good" if v>8 else "Bad"
    if "Debt" in name:
        return "Excellent" if v<0.5 else "Good" if v<1 else "Bad"
    return "Good"

ratio_df = pd.DataFrame(
    [[k,v,grade(k,v)] for k,v in ratios.items()],
    columns=["Ratio","Value","Status"]
)

st.subheader("📊 Financial Ratios")
st.dataframe(ratio_df)

# ---------------- RATIO SUMMARY ----------------
good = (ratio_df["Status"]=="Excellent").sum()
st.info(f"Overall financial strength looks {'Strong' if good>=3 else 'Average'} based on profitability and debt control.")

# ---------------- FORECAST ----------------
st.subheader("📈 Price Projection")

df = data.reset_index()
df["t"] = np.arange(len(df))

X = df[["t"]]
y = df["Close"]

model = LinearRegression().fit(X,y)

future_days = st.slider("Days ahead",30,180,90)

future = np.arange(len(df),len(df)+future_days).reshape(-1,1)
pred = model.predict(future)

pred_df = pd.DataFrame({"Day":range(future_days),"Price":pred})

st.line_chart(pred_df.set_index("Day"))

# ---------------- NEWS ----------------
st.subheader("📰 Latest Stock News")

def fetch_news(q):
    url = f"https://news.google.com/rss/search?q={q}+stock+market"
    feed = feedparser.parse(url)
    news=[]
    for e in feed.entries[:6]:
        published = datetime(*e.published_parsed[:6])
        news.append({
            "title":e.title,
            "date":published.strftime("%d %b %Y %H:%M"),
            "link":e.link
        })
    return news

for n in fetch_news(company):
    st.markdown(f"**{n['title']}**  \n🕒 {n['date']}  \n[Read more]({n['link']})")
