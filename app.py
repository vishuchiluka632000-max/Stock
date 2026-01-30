import streamlit as st
import yfinance as yf
from yfinance import Search
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------- PAGE STYLE ----------------
st.set_page_config(page_title="Stock Analyzer Pro", layout="wide")

st.markdown("""
<style>
.main {background-color:#0e1117;color:white}
.block {background:#161b22;padding:20px;border-radius:16px;margin-bottom:20px}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro — Screener Style")
st.caption("Search by company name like Screener • Full financials • Ratios • Trends")

# ---------------- SMART SEARCH LIKE SCREENER ----------------
query = st.text_input("🔍 Search company name (ex: Reliance, TCS, Infosys)")

symbol = None
company_name = None

if query:
    results = Search(query, max_results=10).quotes
    if results:
        options = {f"{r['shortname']} ({r['symbol']})": r['symbol'] for r in results}
        selected = st.selectbox("Select company", list(options.keys()))
        symbol = options[selected]
        company_name = selected

if not symbol:
    st.info("Start typing company name to search")
    st.stop()

# ---------------- DATA ----------------
def load(symbol):
    s = yf.Ticker(symbol)
    return s.history(period="5y"), s.info, s.financials, s.balance_sheet, s.cashflow

price, info, income, balance, cashflow = load(symbol)

if price.empty:
    st.error("No data found")
    st.stop()

# ---------------- TECHNICALS ----------------
price['MA50'] = price['Close'].rolling(50).mean()
price['MA200'] = price['Close'].rolling(200).mean()

# ---------------- PERFORMANCE ----------------
def CAGR(df):
    return ((df['Close'].iloc[-1]/df['Close'].iloc[0])**(252/len(df))-1)*100

# ---------------- KEY METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", f"₹{price['Close'].iloc[-1]:.2f}")
col2.metric("CAGR", f"{CAGR(price):.2f}%")
col3.metric("52W High", f"₹{price['Close'].tail(252).max():.2f}")
col4.metric("Market Cap", f"₹{round((info.get('marketCap',0)/1e12),2)}T")

# ---------------- PRICE CHART ----------------
st.plotly_chart(px.line(price, y="Close", title=f"{company_name} Price"), use_container_width=True)
st.plotly_chart(px.line(price, y=["Close","MA50","MA200"], title="Trend Averages"), use_container_width=True)

# ---------------- FINANCIAL REPORT ----------------
st.subheader("📑 Financial Statements")

tab1, tab2, tab3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

with tab1:
    st.dataframe(income)
with tab2:
    st.dataframe(balance)
with tab3:
    st.dataframe(cashflow)

# ---------------- RATIOS ----------------
def safe(v): return round(v,2) if v else 0

ratios = pd.DataFrame({
    "Metric": [
        "PE Ratio",
        "ROE %",
        "Profit Margin %",
        "Debt to Equity",
        "Current Ratio",
        "Revenue Growth"
    ],
    "Value": [
        safe(info.get("trailingPE")),
        safe(info.get("returnOnEquity",0)*100),
        safe(info.get("profitMargins",0)*100),
        safe(info.get("debtToEquity")),
        safe(info.get("currentRatio")),
        safe(info.get("revenueGrowth",0)*100)
    ]
})

st.subheader("📊 Financial Ratios")
st.table(ratios)

# ---------------- FORECAST ----------------
reset = price.reset_index()
reset['t'] = np.arange(len(reset))
X = reset[['t']]
y = reset['Close']

model = LinearRegression().fit(X,y)
future = np.arange(len(reset), len(reset)+180).reshape(-1,1)
pred = model.predict(future)

forecast_df = pd.DataFrame({"Day":range(180),"Predicted Price":pred})
st.plotly_chart(px.line(forecast_df, x="Day", y="Predicted Price", title="6 Month Forecast"), use_container_width=True)

# ---------------- HEALTH SCORE ----------------
score = 0
if price['MA50'].iloc[-1] > price['MA200'].iloc[-1]: score += 40
if CAGR(price) > 12: score += 30
if info.get("trailingPE",50) < 25: score += 30

st.subheader("📌 Stock Health Score")
st.progress(score/100)
st.write(f"{score}/100")

# ---------------- SIMILAR STOCKS ----------------
st.subheader("🏭 Similar Companies (same sector)")
st.write(info.get("sector", "Sector data not available"))
