import streamlit as st
import yfinance as yf
from yfinance import Search
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import feedparser
from datetime import datetime

# ---------------- BACKGROUND IMAGE ----------------
st.set_page_config(page_title="Stock Analyzer Pro", layout="wide")

BACKGROUND_URL = "https://images.unsplash.com/photo-1640158616004-3c8fbb3bde43?auto=format&fit=crop&w=1600&q=80"

st.markdown(f"""
<style>
.stApp {{
    background-image: url('{BACKGROUND_URL}');
    background-size: cover;
    background-attachment: fixed;
}}
.block-container {{background:rgba(14,17,23,0.92);padding:2rem;border-radius:20px}}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")
st.caption("Professional stock fundamentals + technical + news dashboard")

# ---------------- SMART SEARCH ----------------
query = st.text_input("🔍 Search company name", placeholder="Reliance, TCS, Infosys, Apple...")

symbol = None
company = None

if query:
    results = Search(query, max_results=15).quotes
    if results:
        options = {f"{r.get('shortname')} ({r.get('symbol')})": r.get('symbol') for r in results if r.get('symbol')}
        selected = st.selectbox("Select company", list(options.keys()))
        symbol = options[selected]
        company = selected.split("(")[0].strip()

if not symbol:
    st.stop()

# ---------------- LOAD DATA ----------------
def load(sym):
    s = yf.Ticker(sym)
    return s.history(period="max"), s.info, s.financials, s.balance_sheet, s.cashflow

price, info, income, balance, cashflow = load(symbol)

if price.empty:
    st.error("No data available")
    st.stop()

# ---------------- TIME RANGE ----------------
periods = {"1D":1,"1W":7,"10D":10,"1M":30,"3M":90,"6M":180,"9M":270,"1Y":365,"3Y":1095,"5Y":1825,"ALL":None}
cols = st.columns(len(periods))
selected = "ALL"
for i,k in enumerate(periods):
    if cols[i].button(k): selected = k

if selected != "ALL":
    price_f = price.tail(periods[selected])
else:
    price_f = price

# ---------------- CANDLESTICK ----------------
fig = go.Figure(data=[go.Candlestick(x=price_f.index,open=price_f['Open'],high=price_f['High'],low=price_f['Low'],close=price_f['Close'])])

market_cap_cr = info.get('marketCap',0)/1e7
fig.add_annotation(text=f"Market Cap: ₹{market_cap_cr:,.0f} Cr",xref="paper",yref="paper",x=1,y=1.15,showarrow=False)
fig.update_layout(title=f"{company} Price Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------- FINANCIAL STATEMENTS ----------------
st.subheader("📑 Financial Statements (₹ Crores)")
fmt = lambda x: f"{x/1e7:,.2f}" if pd.notnull(x) else ""
income_df = income.applymap(fmt)
balance_df = balance.applymap(fmt)
cashflow_df = cashflow.applymap(fmt)

t1,t2,t3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])
with t1: st.dataframe(income_df)
with t2: st.dataframe(balance_df)
with t3: st.dataframe(cashflow_df)

# ---------------- RATIOS + GRADING ----------------
def grade(val, good, excellent):
    if val >= excellent: return "Excellent"
    if val >= good: return "Good"
    return "Bad"

roe = info.get('returnOnEquity',0)*100
profit_margin = info.get('profitMargins',0)*100
debt_eq = info.get('debtToEquity',0)
pe = info.get('trailingPE',0)

ratios = pd.DataFrame({
    "Ratio": ["PE Ratio","ROE %","Net Margin %","Debt to Equity"],
    "Value": [round(pe,2),round(roe,2),round(profit_margin,2),round(debt_eq,2)],
    "Status": [
        grade(25-pe,5,10),
        grade(roe,12,18),
        grade(profit_margin,10,20),
        grade(2-debt_eq,0.5,1)
    ]
})

st.subheader("📊 Financial Ratios")
st.table(ratios)

ratio_summary = f"""
ROE of {roe:.2f}% shows capital efficiency. 
Profit margin at {profit_margin:.2f}% reflects operational strength. 
Debt to Equity of {debt_eq:.2f} indicates leverage risk. 
Overall financial quality appears {'strong' if roe>15 and debt_eq<1 else 'moderate'}.
"""
st.info(ratio_summary)

# ---------------- NEWS FEEDS ----------------
st.subheader("📰 Latest Stock News")

st.markdown("### 🇮🇳 Indian News")n
indian_feeds = [
    "https://indianexpress.com/section/business/feed/",
    "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
    "https://www.indiatoday.in/rss/1206514"
]

for url in indian_feeds:
    feed = feedparser.parse(url)
    for entry in feed.entries[:2]:
        date = entry.get('published','')
        summary = entry.get('summary','')[:150]
        st.markdown(f"**{entry.title}**")
        st.caption(f"{date}")
        st.write(summary + "...")
        st.caption(entry.link)

st.markdown("### 🌍 Global News")n
global_feeds = [
    "https://www.reuters.com/rssFeed/businessNews",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
]

for url in global_feeds:
    feed = feedparser.parse(url)
    for entry in feed.entries[:2]:
        date = entry.get('published','')
        summary = entry.get('summary','')[:150]
        st.markdown(f"**{entry.title}**")
        st.caption(f"{date}")
        st.write(summary + "...")
        st.caption(entry.link)

# ---------------- FORECAST ----------------
reset = price_f.reset_index()
reset['t'] = np.arange(len(reset))
X = reset[['t']]
y = reset['Close']
model = LinearRegression().fit(X,y)
future = np.arange(len(reset), len(reset)+180).reshape(-1,1)
pred = model.predict(future)

forecast_df = pd.DataFrame({"Day":range(180),"Predicted Price":pred})
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_df['Day'], y=forecast_df['Predicted Price']))
fig2.update_layout(title="6 Month Projection")
st.plotly_chart(fig2, use_container_width=True)
