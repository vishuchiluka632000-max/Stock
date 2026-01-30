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

# ---------------- SAFE COMPANY SEARCH ----------------

companies = {
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

results = [c for c in companies if query.lower() in c.lower()]

if not results:
    st.stop()

company = st.selectbox("Select company", results)
symbol = companies[company]

# ---------------- MARKET DATA ----------------

stock = yf.Ticker(symbol)
df = stock.history(period="max")

if df.empty:
    st.error("Market data unavailable")
    st.stop()

info = stock.info
df.reset_index(inplace=True)

# ---------------- TIME RANGE ----------------

ranges = {
    "1D":1,"1W":5,"10D":10,"1M":22,"3M":66,"6M":132,
    "1Y":252,"3Y":756,"5Y":1260,"ALL":len(df)
}

period = st.radio("Range", list(ranges.keys()), horizontal=True)
plot_df = df.tail(ranges[period])

# ---------------- CANDLE CHART ----------------

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

# ---------------- FINANCIAL STATEMENTS ----------------

def crore(df):
    return df.applymap(lambda x: f"{x/1e7:.2f} Cr" if pd.notna(x) else "")

st.subheader("📑 Financial Statements")
a,b,c = st.tabs(["Income","Balance Sheet","Cash Flow"])

with a: st.dataframe(crore(stock.financials))
with b: st.dataframe(crore(stock.balance_sheet))
with c: st.dataframe(crore(stock.cashflow))

# ---------------- FINANCIAL RATIOS ----------------

ratios = {
    "PE Ratio": info.get("trailingPE"),
    "PB Ratio": info.get("priceToBook"),
    "ROE %": info.get("returnOnEquity",0)*100,
    "Profit Margin %": info.get("profitMargins",0)*100,
    "Debt to Equity": info.get("debtToEquity"),
    "Current Ratio": info.get("currentRatio"),
    "Revenue Growth %": info.get("revenueGrowth",0)*100
}

def grade(v, good, ok):
    if v is None: return "N/A"
    if v >= good: return "Excellent"
    if v >= ok: return "Good"
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

strong = (ratio_df["Health"]=="Excellent").sum()
st.success(f"{strong}/{len(ratio_df)} metrics strong — company fundamentals look {'strong' if strong>=4 else 'average'}")

# ---------------- PRICE FORECAST ----------------

df["t"] = range(len(df))
X = df[["t"]]
y = df["Close"]

model = LinearRegression().fit(X,y)

future = np.arange(len(df),len(df)+180).reshape(-1,1)
pred = model.predict(future)

forecast = pd.DataFrame({"Day":range(180),"Price":pred})

st.subheader("📈 6 Month Price Projection")
st.line_chart(forecast.set_index("Day"))

# ---------------- NEWS (STABLE) ----------------

st.subheader("📰 Latest Stock News")

def fetch_news(q):
    feed = feedparser.parse(
        f"https://news.google.com/rss/search?q={q}+stock+market"
    )
    articles=[]
    for e in feed.entries[:6]:
        try:
            r = requests.get(e.link, timeout=5)
            soup = BeautifulSoup(r.text,"html.parser")
            text = " ".join(p.text for p in soup.find_all("p")[:4])
            img = soup.find("img")
            articles.append((e.title,e.published,text[:250],img["src"] if img else None,e.link))
        except:
            pass
    return articles

for t,d,s,i,l in fetch_news(company):
    if i: st.image(i,width=260)
    st.markdown(f"**{t}**")
    st.caption(d)
    st.write(s+"...")
    st.markdown(f"[Read more]({l})")
    st.divider()
