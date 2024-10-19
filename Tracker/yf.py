import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# List of stock tickers
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'LT.NS', 'TATAMOTORS.NS', 
           'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'TITAN.NS', 'WIPRO.NS', 'ICICIBANK.NS', 
           'SBIN.NS', 'HINDUNILVR.NS', 'ITC.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 
           'ADANIENT.NS', 'ASIANPAINT.NS', 'CIPLA.NS']

def predict_arima(data, periods):
    # Fit ARIMA model
    model = ARIMA(data['Close'], order=(5,1,0))  # ARIMA(p,d,q), here (5,1,0) is a typical configuration
    model_fit = model.fit()
    
    # Predict future prices
    forecast = model_fit.forecast(steps=periods)
    
    return forecast

# Sidebar: Select stock
selected_stock = st.sidebar.selectbox('Select Stock:', tickers)

# Sidebar: Select prediction period
prediction_days = st.sidebar.slider('Select number of days to predict:', min_value=1, max_value=365, value=30)

# Fetch historical data
st.write(f"### Historical Data and Prediction for {selected_stock}")
data = fetch_stock_data(selected_stock)

# Plot historical stock data
st.write("Historical Closing Prices:")
st.line_chart(data['Close'])

# Apply ARIMA model and predict future prices
forecast = predict_arima(data, prediction_days)

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

# Initialize portfolio in session_state if it doesn't exist
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=['Stock', 'Quantity', 'Total Price', 'Size', 'Sector', 'Percentage'])

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="5y")  # Fetch 5 years of data
    return hist

# Sidebar: Select stock and enter quantity
selected_stock = st.sidebar.selectbox('Select Stock:', tickers)
quantity = st.sidebar.number_input('Enter Quantity:', min_value=1, value=1)

# Add button
if st.sidebar.button('Add to Portfolio'):
    if selected_stock:
        stock_data = fetch_stock_data(selected_stock)
        ltp = stock_data['Close'].iloc[-1]  # Last traded price from Yahoo Finance
        
        if ltp:
            total_price = ltp * quantity
            stock_name = selected_stock.split('.')[0]
            
            # Check if stock is already in the portfolio
            if stock_name in st.session_state.portfolio_df['Stock'].values:
                index = st.session_state.portfolio_df[st.session_state.portfolio_df['Stock'] == stock_name].index[0]
                st.session_state.portfolio_df.at[index, 'Quantity'] += quantity
                st.session_state.portfolio_df.at[index, 'Total Price'] += total_price
            else:
                # Create new row
                new_row = pd.DataFrame({
                    'Stock': [stock_name],
                    'Quantity': [quantity],
                    'Total Price': [total_price],
                    'Size': ['Large-Cap'],  # Size can be modified according to logic
                    'Sector': ['Unknown'],  # You can map sectors if needed
                    'Percentage': [0.0]
                })
                
                # Add new row to portfolio
                st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, new_row], ignore_index=True)
            
            # Recalculate percentages
            portfolio_value = st.session_state.portfolio_df['Total Price'].sum()
            st.session_state.portfolio_df['Percentage'] = (st.session_state.portfolio_df['Total Price'] / portfolio_value) * 100
            
            st.sidebar.success(f'Added {quantity} shares of {stock_name} to portfolio.')
        else:
            st.sidebar.error(f"Failed to add {selected_stock} to portfolio due to missing data.")

# Display portfolio
st.write("### Current Portfolio")
if st.session_state.portfolio_df.empty:
    st.write("Portfolio is empty.")
else:
    st.write(st.session_state.portfolio_df[['Stock', 'Quantity', 'Total Price', 'Percentage']])
    
    total_value = st.session_state.portfolio_df['Total Price'].sum()
    st.write(f"**Total Portfolio Value: ₹{total_value:,.2f}**")

# Display portfolio distribution pie charts
if not st.session_state.portfolio_df.empty:
    st.write("### Portfolio Distribution by Stock")
    plt.figure(figsize=(4, 4))
    plt.pie(st.session_state.portfolio_df['Total Price'], labels=st.session_state.portfolio_df['Stock'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(plt)
