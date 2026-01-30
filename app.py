import streamlit as st
import yfinance as yf
from yfinance import Search
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

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
st.caption("Screener + TradingView style analysis for beginners & pros")

# ---------------- SMART SEARCH ----------------
query = st.text_input("🔍 Search company name", placeholder="Type: Reliance, TCS, Infosys...")

symbol = None
company = None

if query:
    results = Search(query, max_results=15).quotes
    if results:
        options = {f"{r.get('shortname','')} ({r.get('symbol','')})": r.get('symbol') for r in results if r.get('symbol')}
        selected = st.selectbox("Select company", list(options.keys()))
        symbol = options[selected]
        company = selected

if not symbol:
    st.stop()

# ---------------- DATA ----------------
def load(sym):
    s = yf.Ticker(sym)
    return s.history(period="5y"), s.info, s.financials, s.balance_sheet, s.cashflow

price, info, income, balance, cashflow = load(symbol)

if price.empty:
    st.error("No data available")
    st.stop()

# ---------------- NUMBER FORMAT ----------------
def fmt(x):
    try:
        return f"₹{x:,.0f}"
    except:
        return x

# ---------------- CANDLESTICK CHART ----------------
fig = go.Figure(data=[go.Candlestick(
    x=price.index,
    open=price['Open'],
    high=price['High'],
    low=price['Low'],
    close=price['Close']
)])

fig.update_layout(title=f"{company} Price Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------- PERFORMANCE ----------------
def CAGR(df):
    return ((df['Close'].iloc[-1]/df['Close'].iloc[0])**(252/len(df))-1)*100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", f"₹{price['Close'].iloc[-1]:.2f}")
col2.metric("CAGR", f"{CAGR(price):.2f}%")
col3.metric("52W High", f"₹{price['Close'].tail(252).max():.2f}")
col4.metric("Market Cap", fmt(info.get('marketCap',0)))

# ---------------- FINANCIAL STATEMENTS ----------------
st.subheader("📑 Financial Statements (Absolute Values)")

income_df = income.applymap(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
balance_df = balance.applymap(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
cashflow_df = cashflow.applymap(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

t1, t2, t3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with t1:
    st.dataframe(income_df)
with t2:
    st.dataframe(balance_df)
with t3:
    st.dataframe(cashflow_df)

# ---------------- ADVANCED RATIOS ----------------
def safe(v): return round(v,2) if v else 0

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
        safe(info.get('returnOnAssets',0)*100),
        safe(info.get('profitMargins',0)*100),
        safe(info.get('debtToEquity')),
        safe(info.get('currentRatio')),
        safe(info.get('revenueGrowth',0)*100),
        safe(info.get('earningsGrowth',0)*100)
    ]
})

st.subheader("📊 Key Financial Ratios (Beginner Friendly)")
st.table(ratios)

# ---------------- HEALTH SCORE ----------------
score = 0
if CAGR(price) > 12: score += 25
if ratios.iloc[2,1] > 15: score += 25
if ratios.iloc[5,1] < 1: score += 25
if ratios.iloc[0,1] and ratios.iloc[0,1] < 25: score += 25

st.subheader("📌 Stock Health Score")
st.progress(score/100)
st.write(f"{score}/100")

# ---------------- SIMILAR STOCKS ----------------
sector = info.get('sector','')

st.subheader("🏭 Similar Companies")

if sector:
    peer_results = Search(sector, max_results=10).quotes
    peers = [p.get('shortname') for p in peer_results if p.get('shortname')][:6]
    st.write(", ".join(peers))
else:
    st.write("Sector information not available")

# ---------------- MINI EDUCATION PANEL ----------------
st.subheader("📘 Stock Insight for Beginners")

article = f"""
{company} currently shows a PE ratio of {ratios.iloc[0,1]}, which suggests the market's expectation of future growth.

ROE at {ratios.iloc[2,1]}% indicates how efficiently the company uses shareholder money.

Debt to Equity of {ratios.iloc[5,1]} shows financial risk level.

Overall health score of {score}/100 suggests the stock is {'strong' if score>70 else 'average' if score>40 else 'risky'} for long-term investors.
"""

st.info(article)

# ---------------- FORECAST ----------------
reset = price.reset_index()
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
