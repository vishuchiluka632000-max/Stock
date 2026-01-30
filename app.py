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
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")

# ---------------- SAFE COMPANY SEARCH ----------------

stocks = {
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

query = st.text_input("Search stock name")

matches = [k for k in stocks if query.lower() in k.lower()]

if not matches:
    st.stop()

company = st.selectbox("Select company", matches)
symbol = stocks[company]

# ---------------- DATA ----------------

ticker = yf.Ticker(symbol)
df = ticker.history(period="max")

if df.empty:
    st.error("No data found")
    st.stop()

info = ticker.info
df.reset_index(inplace=True)

# ---------------- TIMEFRAME ----------------

ranges = {
    "1D":1,"1W":5,"10D":10,"1M":22,"3M":66,
    "6M":132,"9M":198,"1Y":252,"3Y":756,
    "5Y":1260,"ALL":len(df)
}

period = st.radio("Chart Range", list(ranges.keys()), horizontal=True)
data = df.tail(ranges[period])

# ---------------- CANDLESTICK ----------------

fig = go.Figure(go.Candlestick(
    x=data["Date"],
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
))

fig.update_layout(template="plotly_dark", height=520)
st.plotly_chart(fig, use_container_width=True)

market_cap = round(info.get("marketCap",0)/1e7,2)
st.markdown(f"### Market Cap: ₹{market_cap} Cr")

# ---------------- FINANCIAL STATEMENTS ----------------

def to_crore(d):
    return d.applymap(lambda x: f"{x/1e7:.2f} Cr" if pd.notna(x) else "")

st.subheader("📑 Financial Statements")

t1,t2,t3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with t1: st.dataframe(to_crore(ticker.financials))
with t2: st.dataframe(to_crore(ticker.balance_sheet))
with t3: st.dataframe(to_crore(ticker.cashflow))

# ---------------- RATIOS ----------------

ratios = {
    "PE Ratio":info.get("trailingPE"),
    "PB Ratio":info.get("priceToBook"),
    "ROE %":info.get("returnOnEquity",0)*100,
    "Profit Margin %":info.get("profitMargins",0)*100,
    "Debt to Equity":info.get("debtToEquity"),
    "Current Ratio":info.get("currentRatio"),
    "Revenue Growth %":info.get("revenueGrowth",0)*100
}

def grade(v, high, mid):
    if v is None: return "N/A"
    if v >= high: return "Excellent"
    if v >= mid: return "Good"
    return "Bad"

rows = [
    ["PE Ratio",ratios["PE Ratio"],grade(ratios["PE Ratio"],25,15)],
    ["PB Ratio",ratios["PB Ratio"],grade(ratios["PB Ratio"],5,2)],
    ["ROE %",ratios["ROE %"],grade(ratios["ROE %"],18,12)],
    ["Profit Margin %",ratios["Profit Margin %"],grade(ratios["Profit Margin %"],15,8)],
    ["Debt to Equity",ratios["Debt to Equity"],grade(1-(ratios["Debt to Equity"] or 1),0.6,0.3)],
    ["Current Ratio",ratios["Current Ratio"],grade(ratios["Current Ratio"],2,1)],
    ["Revenue Growth %",ratios["Revenue Growth %"],grade(ratios["Revenue Growth %"],15,8)]
]

ratio_df = pd.DataFrame(rows, columns=["Metric","Value","Health"])

st.subheader("📊 Financial Ratios")
st.table(ratio_df)

good = (ratio_df["Health"]=="Excellent").sum()
st.success(f"{good}/{len(ratio_df)} ratios are strong. Overall fundamentals look {'strong' if good>=4 else 'average'}.")

# ---------------- FUTURE PROJECTION ----------------

df["t"]=range(len(df))
X=df[["t"]]
y=df["Close"]

model=LinearRegression().fit(X,y)
future=np.arange(len(df),len(df)+180).reshape(-1,1)
pred=model.predict(future)

forecast=pd.DataFrame({"Day":range(180),"Price":pred})

st.subheader("📈 6 Month Projection")
st.line_chart(forecast.set_index("Day"))

# ---------------- NEWS (SAFE + STABLE) ----------------

st.subheader("📰 Latest Stock News")

def fetch_news(q):
    feed = feedparser.parse(
        f"https://news.google.com/rss/search?q={q}+stock+market"
    )
    news=[]
    for e in feed.entries[:6]:
        try:
            r=requests.get(e.link,timeout=5)
            soup=BeautifulSoup(r.text,"html.parser")
            text=" ".join(p.text for p in soup.find_all("p")[:4])
            img=soup.find("img")
            news.append({
                "title":e.title,
                "date":e.published,
                "summary":text[:250],
                "image":img["src"] if img else None,
                "link":e.link
            })
        except:
            pass
    return news

articles = fetch_news(company)

for a in articles:
    if a["image"]:
        st.image(a["image"], width=260)
    st.markdown(f"**{a['title']}**")
    st.caption(a["date"])
    st.write(a["summary"]+"...")
    st.markdown(f"[Read full article]({a['link']})")
    st.divider()
