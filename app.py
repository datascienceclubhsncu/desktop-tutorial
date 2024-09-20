import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv(r"C:\Users\shreay\OneDrive\Desktop\Stocks\master_data.csv")


# Enforcing data types
df['LTP'] = df['LTP'].astype(float)

# Initialize portfolio in session_state if it doesn't exist
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=['Stock', 'Quantity', 'Total Price', 'Size', 'Sector', 'Percentage'])

# Sidebar: Select Size and Sub-Size
selected_size = st.sidebar.selectbox('Select Size:', ['Large-Cap', 'Mid-Cap', 'Small-Cap'])
selected_subsize = st.sidebar.selectbox('Select Sub-Size:', ['All', 'PSU', 'TATA', 'ADANI'])

# Filter stocks based on the selected size and sub-size
if selected_subsize == 'All':
    filtered_stocks = df[df['Size'] == selected_size]['SYMBOL'].tolist()
else:
    filtered_stocks = df[(df['Size'] == selected_size) & (df['Sub-Size'] == selected_subsize)]['SYMBOL'].tolist()

# Dropdown to select stock
selected_stock = st.sidebar.selectbox('Select Stock:', filtered_stocks)

# Quantity input
quantity = st.sidebar.number_input('Enter Quantity:', min_value=1, value=1)

# Add button
if st.sidebar.button('Add to Portfolio'):
    if selected_stock:
        # Get the LTP (Last Traded Price) from the CSV data
        ltp = df[df['SYMBOL'] == selected_stock]['LTP'].values[0]  # Get LTP for selected stock
        
        if ltp is not None:
            total_price = ltp * quantity
            stock_size = df[df['SYMBOL'] == selected_stock]['Size'].values[0]
            stock_sector = df[df['SYMBOL'] == selected_stock]['Sector'].values[0]
            
            # Check if the stock is already in the portfolio
            if selected_stock in st.session_state.portfolio_df['Stock'].values:
                index = st.session_state.portfolio_df[st.session_state.portfolio_df['Stock'] == selected_stock].index[0]
                st.session_state.portfolio_df.at[index, 'Quantity'] += quantity
                st.session_state.portfolio_df.at[index, 'Total Price'] += total_price
            else:
                # Create a new row for the stock
                new_row = pd.DataFrame({
                    'Stock': [selected_stock],
                    'Quantity': [quantity],
                    'Total Price': [total_price],
                    'Size': [stock_size],
                    'Sector': [stock_sector],
                    'Percentage': [0.0]
                })
                
                # Use pd.concat to add the new row to portfolio_df in session_state
                st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, new_row], ignore_index=True)
            
            # Recalculate percentage of total portfolio
            portfolio_value = st.session_state.portfolio_df['Total Price'].sum()
            if portfolio_value > 0:
                st.session_state.portfolio_df['Percentage'] = (st.session_state.portfolio_df['Total Price'] / portfolio_value) * 100
            
            st.sidebar.success(f'Added {quantity} shares of {selected_stock} to portfolio.')
        else:
            st.sidebar.error(f"Failed to add {selected_stock} to portfolio due to missing LTP data.")

# Display portfolio
st.write("### Current Portfolio")
if st.session_state.portfolio_df.empty:
    st.write("Portfolio is empty.")
else:
    # Display the portfolio DataFrame
    st.write(st.session_state.portfolio_df[['Stock', 'Quantity', 'Total Price', 'Percentage']])
    
    # Calculate and display total portfolio value
    total_value = st.session_state.portfolio_df['Total Price'].sum()
    st.write(f"**Total Portfolio Value: ₹{total_value:,.2f}**")  # Format the total value with commas and 2 decimal points

# Display portfolio distribution pie charts
if not st.session_state.portfolio_df.empty:
    # Portfolio Distribution by Size
    st.write("### Portfolio Distribution by Size")
    size_distribution = st.session_state.portfolio_df.groupby('Size')['Total Price'].sum()
    plt.figure(figsize=(4, 4))
    plt.pie(size_distribution, labels=size_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(plt)

    # Portfolio Distribution by Stock
    st.write("### Portfolio Distribution by Stock")
    plt.figure(figsize=(4, 4))
    plt.pie(st.session_state.portfolio_df['Total Price'], labels=st.session_state.portfolio_df['Stock'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(plt)

    # Portfolio Distribution by Sector
    st.write("### Portfolio Distribution by Sector")
    sector_distribution = st.session_state.portfolio_df.groupby('Sector')['Total Price'].sum()
    plt.figure(figsize=(4, 4))
    plt.pie(sector_distribution, labels=sector_distribution.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(plt)