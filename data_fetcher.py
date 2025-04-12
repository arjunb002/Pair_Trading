import yfinance as yf
import pandas as pd

class DataFetcher:
    def get_stock_data(self, ticker, period='1y'):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}. Please try again.")
                
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = 0
                    
            return data
        except Exception as e:
            raise ValueError(f"Error fetching data for {ticker}: {str(e)}")
    
    def validate_ticker(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info is not None and len(info) > 0
        except:
            return False