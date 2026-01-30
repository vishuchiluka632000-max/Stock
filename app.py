import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, feedparser
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

st.set_page_config("Stock Analyzer Pro", layout="wide")

# ------------------ UI STYLE ------------------

st.markdown("""
<style>
body {background:#0e1117;color:white}
.card {background:#161b22;padding:18px;border-radius:14px}
button {border-radius:12px}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analyzer Pro")

# ------------------ SMART SEARCH ------------------

query = st.text_input("Search stock name or symbol")

def search_stock(q):
    if not q:
        return []
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={q}"
    return requests.get(url).json()["quotes"][:6]

results = search_stock(query)

if results:
    selected = st.selectbox(
        "Select company",
        results,
        format_func=lambda x: f"{x['shortname']} ({x['symbol']})"
    )
    symbol = selected["symbol"]
else:
    st.stop()

# ------------------ DATA ------------------

stock = yf.Ticker(symbol)
df = stock.history(period="max")
info = stock.info

df.reset_index(inplace=True)

# ------------------ TIMEFRAME ------------------

ranges = {
    "1D":1, "1W":5, "10D":10,
    "1M":22,"3M":66,"6M":132,
    "1Y":252,"3Y":756,"5Y":1260,"ALL":len(df)
}

choice = st.radio("Time Range", list(ranges.keys()), horizontal=True)
plot_df = df.tail(ranges[choice])

# ------------------ CANDLE CHART ------------------

fig = go.Figure(data=[go.Candlestick(
    x=plot_df["Date"],
    open=plot_df["Open"],
    high=plot_df["High"],
    low=plot_df["Low"],
    close=plot_df["Close"]
)])

fig.update_layout(
    template="plotly_dark",
    title=f"{info.get('shortName')} Price Chart",
    height=520
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"### Market Cap: ₹{round(info.get('marketCap',0)/1e7,2)} Cr"
)

# ------------------ FINANCIAL STATEMENTS ------------------

def format_cr(df):
    return df.applymap(lambda x: f"{x/1e7:.2f} Cr" if pd.notna(x) else "")

st.subheader("📑 Financial Statements")

tab1,tab2,tab3 = st.tabs(["Income","Balance Sheet","Cash Flow"])

with tab1:
    st.dataframe(format_cr(stock.financials))
with tab2:
    st.dataframe(format_cr(stock.balance_sheet))
with tab3:
    st.dataframe(format_cr(stock.cashflow))

# ------------------ RATIOS ------------------

ratios = {
    "PE Ratio": info.get("trailingPE"),
    "PB Ratio": info.get("priceToBook"),
    "ROE %": info.get("returnOnEquity",0)*100,
    "Profit Margin %": info.get("profitMargins",0)*100,
    "Debt to Equity": info.get("debtToEquity"),
    "Current Ratio": info.get("currentRatio"),
    "Revenue Growth %": info.get("revenueGrowth",0)*100
}

def grade(val, good, bad):
    if val is None: return "N/A"
    if val >= good: return "Excellent"
    if val >= bad: return "Good"
    return "Bad"

rows = []

rows.append(["PE Ratio", ratios["PE Ratio"], grade(ratios["PE Ratio"],25,15)])
rows.append(["PB Ratio", ratios["PB Ratio"], grade(ratios["PB Ratio"],5,2)])
rows.append(["ROE %", ratios["ROE %"], grade(ratios["ROE %"],18,12)])
rows.append(["Profit Margin %", ratios["Profit Margin %"], grade(ratios["Profit Margin %"],15,8)])
rows.append(["Debt to Equity", ratios["Debt to Equity"], grade(1-ratios["Debt to Equity"] if ratios["Debt to Equity"] else 0,0.6,0.3)])
rows.append(["Current Ratio", ratios["Current Ratio"], grade(ratios["Current Ratio"],2,1)])
rows.append(["Revenue Growth %", ratios["Revenue Growth %"], grade(ratios["Revenue Growth %"],15,8)])

ratio_df = pd.DataFrame(rows, columns=["Metric","Value","Status"])

st.subheader("📊 Financial Ratios")
st.table(ratio_df)

good = ratio_df["Status"].value_counts().get("Excellent",0)
st.info(f"Summary: {good} strong metrics out of {len(ratio_df)} — overall {'Strong' if good>=4 else 'Average'} company fundamentals.")

# ------------------ PRICE FORECAST ------------------

st.subheader("📈 Price Projection")

df["t"]=range(len(df))
X=df[["t"]]
y=df["Close"]

model=LinearRegression().fit(X,y)

future_t=np.arange(len(df),len(df)+180).reshape(-1,1)
pred=model.predict(future_t)

forecast=pd.DataFrame({"Day":range(180),"Price":pred})

st.line_chart(forecast.set_index("Day"))

# ------------------ NEWS ------------------

st.subheader("📰 Stock News")

def fetch_news(q):
    feeds={
        "Indian":f"https://news.google.com/rss/search?q={q}+india+stock",
        "Global":f"https://news.google.com/rss/search?q={q}+stock+market"
    }

    data={}

    for k,u in feeds.items():
        f=feedparser.parse(u)
        items=[]
        for e in f.entries[:4]:
            try:
                r=requests.get(e.link,timeout=5)
                soup=BeautifulSoup(r.text,"html.parser")
                text=" ".join(p.text for p in soup.find_all("p")[:6])
                img=soup.find("img")
                items.append({
                    "title":e.title,
                    "date":e.published,
                    "summary":text[:280],
                    "img":img["src"] if img else None,
                    "link":e.link
                })
            except:
                pass
        data[k]=items
    return data

news=fetch_news(info.get("shortName"))

for region,articles in news.items():
    st.markdown(f"## {region} News")
    for n in articles:
        if n["img"]:
            st.image(n["img"],width=280)
        st.markdown(f"**{n['title']}**")
        st.caption(n["date"])
        st.write(n["summary"]+"...")
        st.markdown(f"[Read full story]({n['link']})")
        st.divider()
