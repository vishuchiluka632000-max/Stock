import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import feedparser
import requests
from newspaper import Article

# ---------------- UI ----------------
st.set_page_config("Stock Analyzer Pro", layout="wide")

def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("bg.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.title("📊 Stock Analyzer Pro")

# ---------------- SEARCH ----------------
@st.cache_data
def load_symbols():
    tickers = yf.Tickers(" ".join([
        "RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ICICIBANK.NS SBIN.NS IEX.NS ITC.NS"
    ]))
    return {t.info.get("longName", t): k for k,t in tickers.tickers.items()}

symbols = load_symbols()
company = st.selectbox("Search Company", list(symbols.keys()))
ticker = symbols[company]

# ---------------- TIME RANGE ----------------
ranges = {
    "1D":"1d","1W":"5d","10D":"10d","1M":"1mo",
    "3M":"3mo","6M":"6mo","9M":"9mo",
    "1Y":"1y","3Y":"3y","5Y":"5y","ALL":"max"
}

cols = st.columns(len(ranges))
period = "1y"
for i,(k,v) in enumerate(ranges.items()):
    if cols[i].button(k):
        period = v

# ---------------- PRICE DATA ----------------
data = yf.download(ticker, period=period, interval="1d")

# ---------------- CHART MODE ----------------
mode = st.radio("Chart Type", ["Line","Candlestick + Volume"], horizontal=True)

fig = go.Figure()

if mode=="Line":
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price"))
else:
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"]
    ))
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", yaxis="y2"))

fig.update_layout(
    yaxis2=dict(overlaying="y", side="right", showgrid=False),
    height=600,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET CAP ----------------
info = yf.Ticker(ticker).info
mcap = info.get("marketCap",0)/1e7
st.markdown(f"### Market Cap: ₹ {mcap:,.0f} Cr")

# ---------------- FINANCIALS ----------------
st.subheader("📑 Financial Statements")

fs = yf.Ticker(ticker)
tabs = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])

for tab,df in zip(tabs,[fs.financials,fs.balance_sheet,fs.cashflow]):
    with tab:
        if df is not None:
            short = df/1e7
            st.dataframe(short.round(2))

# ---------------- RATIOS ----------------
def grade(val, good, ok):
    if val>=good: return "Excellent"
    if val>=ok: return "Good"
    return "Bad"

ratios = {
    "PE": info.get("trailingPE",0),
    "ROE %": info.get("returnOnEquity",0)*100,
    "ROCE %": info.get("returnOnAssets",0)*100,
    "Profit Margin %": info.get("profitMargins",0)*100,
    "Debt/Equity": info.get("debtToEquity",0),
    "Current Ratio": info.get("currentRatio",0)
}

rows=[]
for k,v in ratios.items():
    if "RO" in k or "Margin" in k:
        status = grade(v,15,8)
    elif "PE" in k:
        status = grade(30-v,10,0)
    else:
        status = grade(2-v,0.5,0)

    rows.append([k,round(v,2),status])

ratio_df = pd.DataFrame(rows, columns=["Ratio","Value","Health"])
st.subheader("📊 Financial Ratios")
st.dataframe(ratio_df)

good = ratio_df["Health"].value_counts().to_dict()
st.info(f"Summary: {good}")

# ---------------- FORECAST ----------------
st.subheader("📈 Future Projection")

f_range = st.radio("Projection Range",["3M","6M","1Y"],horizontal=True)
days = {"3M":90,"6M":180,"1Y":365}[f_range]

reset = data.reset_index()
reset["t"]=np.arange(len(reset))
model = LinearRegression().fit(reset[["t"]],reset["Close"])

future = np.arange(len(reset),len(reset)+days).reshape(-1,1)
pred = model.predict(future)

pf = pd.DataFrame({"Day":range(days),"Forecast":pred})

st.line_chart(pf.set_index("Day"))

# ---------------- UPLOAD FS ----------------
st.subheader("📂 Upload Screener Financial File")

file = st.file_uploader("Upload Excel/CSV")
if file:
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    st.dataframe(df)

# ---------------- SIMILAR STOCKS ----------------
sector = info.get("sector","")
st.subheader("🏭 Similar Sector Companies")
st.write(sector)

# ---------------- NEWS ----------------
def fetch_news(q):
    feeds=[
        f"https://news.google.com/rss/search?q={q}+india+stock",
        f"https://news.google.com/rss/search?q={q}+global+stock"
    ]
    articles=[]
    for url in feeds:
        feed=feedparser.parse(url)
        for e in feed.entries[:4]:
            try:
                art=Article(e.link)
                art.download(); art.parse()
                articles.append({
                    "title":e.title,
                    "date":e.published,
                    "summary":art.text[:300],
                    "image":art.top_image,
                    "link":e.link
                })
            except:
                pass
    return articles

st.subheader("📰 Latest News")

news = fetch_news(company)

for n in news:
    if n["image"]:
        st.image(n["image"], width=300)
    st.markdown(f"**{n['title']}**")
    st.caption(n["date"])
    st.write(n["summary"]+"...")
    st.markdown(f"[Read more]({n['link']})")
    st.divider()
