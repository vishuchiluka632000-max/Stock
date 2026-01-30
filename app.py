import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import feedparser
from datetime import datetime

st.set_page_config("Stock Analyzer Pro", layout="wide")

st.markdown("""
<style>
body {background:#0e1117;color:white}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")

# ---------------- STOCK SEARCH ----------------

query = st.text_input("Search stock (NSE, BSE, Global)")

def get_suggestions(q):
    if not q:
        return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
    try:
        r = yf.utils.get_json(url)
        return [(x["shortname"], x["symbol"]) for x in r["quotes"][:6]]
    except:
        return []

suggestions = get_suggestions(query)

symbol = None
if suggestions:
    choice = st.selectbox("Select company", suggestions, format_func=lambda x: f"{x[0]} ({x[1]})")
    symbol = choice[1]

if not symbol:
    st.stop()

# ---------------- DATA ----------------

stock = yf.Ticker(symbol)
df = stock.history(period="max")
info = stock.info

# ---------------- RANGE BUTTONS ----------------

ranges = {
    "1D":"1d","1W":"5d","1M":"1mo","3M":"3mo",
    "6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y","ALL":"max"
}

cols = st.columns(len(ranges))
period = "1y"
for i,(k,v) in enumerate(ranges.items()):
    if cols[i].button(k):
        period = v

df = stock.history(period=period)

# ---------------- CHART TYPE ----------------

chart_type = st.radio("Chart", ["Candlestick","Line"], horizontal=True)

fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]
    )
else:
    fig.add_scatter(x=df.index, y=df["Close"], line=dict(color="cyan"))

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET INFO ----------------

mc = info.get("marketCap",0)/1e7
st.write(f"### Market Cap: ₹ {mc:,.0f} Cr")

# ---------------- FINANCIALS ----------------

st.subheader("📑 Financial Statements")

tabs = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with tabs[0]:
    st.dataframe(stock.financials)

with tabs[1]:
    st.dataframe(stock.balance_sheet)

with tabs[2]:
    st.dataframe(stock.cashflow)

# ---------------- RATIOS ----------------

def grade(val, good, bad):
    if val >= good: return "🟢 Excellent"
    if val >= bad: return "🟡 Good"
    return "🔴 Bad"

ratios = [
    ("PE", info.get("trailingPE",0), 25, 40),
    ("ROE %", info.get("returnOnEquity",0)*100, 15, 8),
    ("Profit Margin %", info.get("profitMargins",0)*100, 15, 5),
    ("Debt/Equity", info.get("debtToEquity",0), 1, 2),
    ("Revenue Growth %", info.get("revenueGrowth",0)*100, 10, 5)
]

ratio_df = pd.DataFrame(
    [(n, round(v,2), grade(v,g,b)) for n,v,g,b in ratios],
    columns=["Ratio","Value","Status"]
)

st.subheader("📊 Financial Ratios")
st.dataframe(ratio_df)

good_count = sum(ratio_df["Status"].str.contains("Excellent"))
st.success(f"Overall Strength: {good_count}/{len(ratios)} strong metrics")

# ---------------- FORECAST ----------------

st.subheader("📈 Price Projection")

days = st.selectbox("Projection Range", [30,90,180,365])

df2 = df.reset_index()
df2["t"] = np.arange(len(df2))

X = df2[["t"]]
y = df2["Close"]

model = LinearRegression().fit(X,y)

future_t = np.arange(len(df2), len(df2)+days)
pred = model.predict(future_t.reshape(-1,1))

forecast = pd.DataFrame({
    "Day": range(1,days+1),
    "Predicted Price": pred.flatten()
})

fig2 = go.Figure()
fig2.add_scatter(x=forecast["Day"], y=forecast["Predicted Price"], line=dict(color="orange"))
fig2.update_layout(height=400)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- NEWS ----------------

st.subheader("📰 Latest Stock News")

def fetch_news(q):
    feed = feedparser.parse(
        f"https://news.google.com/rss/search?q={q}+stock+market"
    )
    news = []
    for e in feed.entries[:6]:
        news.append({
            "title": e.title,
            "time": e.published,
            "link": e.link
        })
    return news

news = fetch_news(symbol)

for n in news:
    st.markdown(f"**{n['title']}**")
    st.caption(n["time"])
    st.write(n["link"])
    st.markdown("---")

# ---------------- SIMILAR STOCKS ----------------

sector = info.get("sector","")

if sector:
    st.subheader("🏭 Similar Sector Stocks")
    peers = yf.Ticker(info.get("symbol")).recommendations
    st.write("Explore peers in same sector manually (Yahoo API limitation)")
else:
    st.info("Sector data unavailable")

