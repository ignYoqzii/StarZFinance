import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

data = yf.download("AAPL", start="2020-02-10", end="2025-02-11")
close = pd.DataFrame(data['Close'])

close.to_csv("file.csv")