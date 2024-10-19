import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import plotly.express as px

# List of stock tickers
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'LT.NS', 'TATAMOTORS.NS', 
           'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'TITAN.NS', 'WIPRO.NS', 'ICICIBANK.NS', 
           'SBIN.NS', 'HINDUNILVR.NS', 'ITC.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 
           'ADANIENT.NS', 'ASIANPAINT.NS', 'CIPLA.NS']

# Function to fetch stock data
def fetch_stock_data(ticker, period="5y"):
    try:
        stock_data = yf.Ticker(ticker)
        hist = stock_data.history(period=period)
        if hist.empty:
            st.error("No data found for the selected stock. Please try a different stock.")
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}. Please try again later.")
        st.write(f"Error details: {e}")
        return pd.DataFrame()  # Return empty DataFrame if fetching fails

# Function to apply ARIMA and predict future prices
def predict_arima(data, periods):
    try:
        # Fit ARIMA model
        model = ARIMA(data['Close'], order=(5,1,0))  # ARIMA(p,d,q), here (5,1,0) is a typical configuration
        model_fit = model.fit()
        
        # Predict future prices
        forecast = model_fit.forecast(steps=periods)
        
        return forecast
    except Exception as e:
        st.error(f"Error while predicting using ARIMA model: {e}")
        return pd.Series()  # Return empty series on error

# Sidebar: Select stock
selected_stock = st.sidebar.selectbox('Select Stock for ARIMA Prediction:', tickers)

# Sidebar: Select prediction period
prediction_days = st.sidebar.slider('Select number of days to predict:', min_value=1, max_value=365, value=30)

# Fetch historical data
st.write(f"### Historical Data and Prediction for {selected_stock}")
data = fetch_stock_data(selected_stock)

# Ensure we have data before proceeding
if not data.empty:
    # Plot historical stock data
    st.write("Historical Closing Prices:")
    st.line_chart(data['Close'])
    
    # Apply ARIMA model and predict future prices
    forecast = predict_arima(data, prediction_days)
    
    if not forecast.empty:
        # Prepare dates for forecast
        last_date = data.index[-1]
        forecast_dates = pd.date_range(last_date, periods=prediction_days+1, closed='right')
        
        # Combine historical and forecasted data
        forecast_series = pd.Series(forecast, index=forecast_dates)
        combined_data = pd.concat([data['Close'], forecast_series])
        
        # Plot historical and predicted prices
        st.write(f"Predicted Closing Prices for the next {prediction_days} days:")
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Historical Prices')
        plt.plot(forecast_series.index, forecast_series, label='Predicted Prices', color='red')
        plt.title(f'{selected_stock} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        st.pyplot(plt)
        
        # Display the predicted prices as a table
        st.write(f"Predicted Prices for the next {prediction_days} days:")
        st.dataframe(forecast_series)

# Sector distribution pie chart
st.write("### Portfolio Sector Distribution")
portfolio_df = pd.DataFrame(portfolio_data)

# Create a pie chart using Plotly
sector_distribution = portfolio_df.groupby('Sector').sum()['LTP']
fig = px.pie(sector_distribution, values='LTP', names=sector_distribution.index, title='Sector Distribution in Portfolio')
st.plotly_chart(fig)


