import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Netflix Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)
st.sidebar.title("Dashboard")

days = st.sidebar.slider(
    "Forecast Days",
    7,
    90,
    30
)

show_data = st.sidebar.checkbox(
    "Show Dataset",
    True
)

st.title("📈 Netflix Stock Forecast Dashboard")

st.markdown("""
This dashboard analyzes historical Netflix stock prices and predicts future prices using the *ARIMA Time Series Model*.
""")
st.write("Time Series Forecasting using ARIMA Model")

# Load dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload Stock CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("NFLX.csv")

df = df.drop(['Open','High','Low','Close','Volume'], axis=1)
df['Date'] = pd.to_datetime(df['Date'])
df = df.groupby('Date')['Adj Close'].sum().reset_index()
df.set_index('Date', inplace=True)
col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Price",
    f"${df['Adj Close'].iloc[-1]:.2f}"
)

col2.metric(
    "Highest Price",
    f"${df['Adj Close'].max():.2f}"
)

col3.metric(
    "Lowest Price",
    f"${df['Adj Close'].min():.2f}"
)
st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("Rows:", df.shape[0])

with col2:
    st.write("Columns:", df.shape[1])

st.subheader("Dataset")
if show_data:
    st.subheader("Dataset")
    st.dataframe(df)

# Stock Price Plot
st.subheader("Netflix Stock Price")
fig = px.line(
    df,
    x=df.index,
    y="Adj Close",
    title="Netflix Stock Price"
)

st.plotly_chart(fig, use_container_width=True)

# ADF Test
st.subheader("ADF Test")
result = adfuller(df["Adj Close"])

st.write("ADF Statistic:", result[0])
st.write("P-value:", result[1])

if result[1] > 0.05:
    st.error("Data is NOT Stationary")
else:
    st.success("Data is Stationary")

# Seasonal Decomposition
st.subheader("Seasonal Decomposition")

decomp = seasonal_decompose(df["Adj Close"], period=5)
fig = decomp.plot()
st.pyplot(fig)

# ARIMA Model
model = ARIMA(df["Adj Close"], order=(0,1,0))
fit = model.fit()

df["Prediction"] = fit.predict(start=df.index[0], end=df.index[-1])

st.subheader("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df.index, df["Adj Close"], label="Actual")
ax.plot(df.index, df["Prediction"], label="Prediction")
ax.legend()
st.pyplot(fig)

# Forecast
forecast = fit.forecast(steps=days)

st.subheader("30-Day Forecast")
future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=days
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast Price": forecast.values
})

st.subheader("Forecast")

st.dataframe(forecast_df)

fig = px.line(
    forecast_df,
    x="Date",
    y="Forecast Price",
    title="Future Forecast"
)

st.plotly_chart(fig, use_container_width=True)