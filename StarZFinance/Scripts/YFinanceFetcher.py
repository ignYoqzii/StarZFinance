import yfinance as yf


# Function to fetch stock data from Yahoo Finance API
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    return stock_data


# Function to fetch today's close price of the given tickers (list of stock symbols)
def fetch_close_price(tickers):
    close_prices = {}
    for ticker in tickers:
        data = yf.Ticker(ticker)
        today_data = data.history(period="1d")
        close_prices[ticker] = round((today_data["Close"][0]), 2)
    return close_prices
