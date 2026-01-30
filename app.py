import streamlit as st
import yfinance as yf
from yfinance import Search
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import feedparser
from datetime import timedelta

# ---------------- PAGE STYLE ----------------
st.set_page_config(page_title="Stock Analyzer Pro", layout="wide")

st.markdown("""
<style>
.main {background:#0e1117;color:white}
.card {background:#161b22;padding:20px;border-radius:16px;margin-bottom:20px}
.small {font-size:13px;color:#9ca3af}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro — Full Fundamental Platform")
st.caption("Screener + TradingView style analysis for Indian & global stocks")

# ---------------- SMART SEARCH (NAME MATCH FIXED) ----------------
query = st.text_input("🔍 Search company name", placeholder="Type: Reliance, TCS, Infosys...")

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

# ---------------- DATA ----------------
def load(sym):
    s = yf.Ticker(sym)
    return s.history(period="max"), s.info, s.financials, s.balance_sheet, s.cashflow

price, info, income, balance, cashflow = load(symbol)

if price.empty:
    st.error("No data available")
    st.stop()

# ---------------- HELPERS ----------------
def to_crore(x):
    try:
        return f"₹{x/1e7:,.2f} Cr"
    except:
        return ""

# ---------------- TIME RANGE SELECTOR ----------------
periods = {
    "1D":1,
    "1W":7,
    "10D":10,
    "1M":30,
    "3M":90,
    "6M":180,
    "9M":270,
    "1Y":365,
    "3Y":1095,
    "5Y":1825,
    "ALL":None
}

cols = st.columns(len(periods))
selected_period = "ALL"

for i,(k,v) in enumerate(periods.items()):
    if cols[i].button(k): selected_period = k

if selected_period != "ALL":
    days = periods[selected_period]
    price_filtered = price.tail(days)
else:
    price_filtered = price

# ---------------- CANDLESTICK CHART ----------------
fig = go.Figure(data=[go.Candlestick(
    x=price_filtered.index,
    open=price_filtered['Open'],
    high=price_filtered['High'],
    low=price_filtered['Low'],
    close=price_filtered['Close']
)])

market_cap_cr = info.get('marketCap',0)/1e7

fig.add_annotation(
    text=f"Market Cap: ₹{market_cap_cr:,.0f} Cr",
    xref="paper", yref="paper",
    x=1, y=1.15, showarrow=False
)

fig.update_layout(title=f"{company} Price Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------- FINANCIAL STATEMENTS (SHORT FORMAT) ----------------
st.subheader("📑 Financial Statements (₹ in Crores)")

income_df = income.applymap(lambda x: f"{x/1e7:,.2f}" if pd.notnull(x) else "")
balance_df = balance.applymap(lambda x: f"{x/1e7:,.2f}" if pd.notnull(x) else "")
cashflow_df = cashflow.applymap(lambda x: f"{x/1e7:,.2f}" if pd.notnull(x) else "")

t1, t2, t3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with t1: st.dataframe(income_df)
with t2: st.dataframe(balance_df)
with t3: st.dataframe(cashflow_df)

# ---------------- PROPER RATIO CALCULATION ----------------
def safe(v): return round(v,2) if v and not pd.isna(v) else 0

try:
    ebit = income.loc['Ebit'].iloc[0]
    total_assets = balance.loc['Total Assets'].iloc[0]
    current_liabilities = balance.loc['Current Liabilities'].iloc[0]
    roce = (ebit / (total_assets - current_liabilities)) * 100
except:
    roce = 0

ratios = pd.DataFrame({
    "Ratio": [
        "PE Ratio",
        "PB Ratio",
        "ROE %",
        "ROCE %",
        "Net Profit Margin %",
        "Debt to Equity",
        "Current Ratio",
        "Revenue Growth %",
        "EPS Growth %"
    ],
    "Value": [
        safe(info.get('trailingPE')),
        safe(info.get('priceToBook')),
        safe(info.get('returnOnEquity',0)*100),
        safe(roce),
        safe(info.get('profitMargins',0)*100),
        safe(info.get('debtToEquity')),
        safe(info.get('currentRatio')),
        safe(info.get('revenueGrowth',0)*100),
        safe(info.get('earningsGrowth',0)*100)
    ]
})

st.subheader("📊 Financial Ratios")
st.table(ratios)

# ---------------- RATIO SUMMARY ----------------
summary = f"""
{company} shows a PE of {ratios.iloc[0,1]}, indicating valuation level.

ROE of {ratios.iloc[2,1]}% reflects shareholder return efficiency.

ROCE of {ratios.iloc[3,1]}% measures business capital productivity.

Debt to Equity at {ratios.iloc[5,1]} indicates leverage risk.

Overall, growth and profitability appear {'strong' if ratios.iloc[2,1]>15 else 'moderate' if ratios.iloc[2,1]>8 else 'weak'}.
"""

st.info(summary)

# ---------------- STOCK NEWS (INDIAN SOURCES) ----------------
st.subheader("📰 Latest Stock News")

feeds = [
    "https://indianexpress.com/section/business/feed/",
    "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
    "https://www.indiatoday.in/rss/1206514"
]

for url in feeds:
    feed = feedparser.parse(url)
    for entry in feed.entries[:2]:
        st.markdown(f"**{entry.title}**")
        st.caption(entry.link)

# ---------------- SIMPLE FORECAST ----------------
reset = price_filtered.reset_index()
reset['t'] = np.arange(len(reset))
X = reset[['t']]
y = reset['Close']

model = LinearRegression().fit(X,y)
future = np.arange(len(reset), len(reset)+180).reshape(-1,1)
pred = model.predict(future)

forecast_df = pd.DataFrame({"Day":range(180),"Predicted Price":pred})

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_df['Day'], y=forecast_df['Predicted Price']))
fig2.update_layout(title="6 Month Price Projection")

st.plotly_chart(fig2, use_container_width=True)
