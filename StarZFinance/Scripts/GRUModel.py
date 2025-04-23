import os
import sys
import io
import json
import base64
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional

# Suppress warnings and logging messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL + 1)

# Data and plotting.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Deep learning.
from keras.models import Sequential
from keras.layers import Input, GRU, Dense, Dropout  # Switched from LSTM to GRU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Custom modules.
import YFinanceFetcher as yff
import JSONManager as jsonm

# Custom matplotlib style.
plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


class StockPredictorGRU:
    def __init__(
        self,
        time_step: int = 60,
        feature: str = "Close",
        gru_units: int = 50,
        dropout_rate: float = 0.2,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "adam",
        use_early_stopping: bool = False,
        show_future_actual: bool = False,
        use_sentiment_analysis: bool = False,
        scaling_factor: float = 5,
        plot_window: int = 0,  # Number of days to show in plot, 0 for all data
    ):
        # Hyperparameter assignments
        self.time_step = time_step
        self.feature = feature
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.use_early_stopping = use_early_stopping
        self.show_future_actual = show_future_actual
        self.use_sentiment_analysis = use_sentiment_analysis
        self.scaling_factor = scaling_factor
        self.plot_window = plot_window

        # Variables to hold the built model and the scaler after training
        self.model = None
        self.scaler = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        data_values = data[self.feature].values.reshape(-1, 1).astype(float)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data_values)

        X, y = [], []
        for i in range(self.time_step, len(data_scaled)):
            X.append(data_scaled[i - self.time_step : i])
            y.append(data_scaled[i])
        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                GRU(units=self.gru_units, return_sequences=True),  # GRU layer
                Dropout(self.dropout_rate),
                GRU(units=self.gru_units, return_sequences=False),  # GRU layer
                Dropout(self.dropout_rate),
                Dense(units=1),
            ]
        )
        model.compile(optimizer=self.optimizer, loss="mse")
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        if self.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0,
            )
        else:
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0,
            )

    def predict(self, X_input: np.ndarray, days_to_predict: int = 5) -> np.ndarray:
        predicted_values = []
        for _ in range(days_to_predict):
            predicted = self.model.predict(X_input, verbose=0)
            predicted_values.append(predicted[0])
            X_input = np.concatenate(
                [X_input[:, 1:, :], predicted.reshape(1, 1, -1)], axis=1
            )
        predicted_values = np.array(predicted_values)
        return self.scaler.inverse_transform(predicted_values)

    def adjust_predictions(self, predictions: np.ndarray) -> np.ndarray:
        sentiment_score = 0.5
        predictions[:, 0] += sentiment_score * self.scaling_factor
        return predictions

    def plot_full_timeline(
        self,
        train_data: np.ndarray,
        train_dates: pd.DatetimeIndex,
        train_pred: np.ndarray,
        future_pred: np.ndarray,
        ticker: str,
        future_dates: pd.DatetimeIndex,
        future_actual: Optional[np.ndarray] = None,
    ) -> io.BytesIO:
        plt.figure(figsize=(12, 6))
        if self.plot_window > 0:
            window_start = max(0, len(train_dates) - self.plot_window)
            train_data = train_data[window_start:]
            train_dates = train_dates[window_start:]
            train_pred = train_pred[window_start:]

        plt.plot(
            train_dates,
            train_data.reshape(-1),
            label="Real Stock Prices",
            linewidth=2,
        )

        valid_train_pred = train_pred[~np.isnan(train_pred).any(axis=1)]
        valid_pred_dates = train_dates[-len(valid_train_pred) :]
        plt.plot(
            valid_pred_dates,
            valid_train_pred.reshape(-1),
            label="Training Predictions",
            linestyle="--",
            linewidth=2,
        )

        plt.plot(
            future_dates,
            future_pred.reshape(-1),
            label="Future Predictions",
            linewidth=2,
        )

        if future_actual is not None:
            plt.plot(
                future_dates,
                future_actual.reshape(-1),
                label="Real Future Prices",
                linestyle="--",
            )

        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Stock Price", fontsize=12)
        window_text = f" (Last {self.plot_window} days)" if self.plot_window else ""
        plt.title(f"{ticker} - Stock Price Prediction{window_text}", fontsize=14)
        plt.legend()
        plt.grid(True)

        total_days = (train_dates[-1] - train_dates[0]).days + len(future_dates)
        if (self.plot_window > 0 and self.plot_window <= 90) or (
            self.plot_window == 0 and total_days <= 90
        ):
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
        else:
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        buf = io.BytesIO()
        canvas = FigureCanvas(plt.gcf())
        canvas.print_png(buf)
        plt.close()
        buf.seek(0)
        return buf

    def run(
        self, ticker: str, start_date: str, end_date: str, days_to_predict: int = 5
    ) -> io.BytesIO:
        stock_data = yff.fetch_stock_data(ticker, start_date, end_date)
        if len(stock_data) < self.time_step + days_to_predict:
            raise ValueError(
                f"Not enough data. Need at least {self.time_step + days_to_predict} days."
            )

        if self.show_future_actual:
            training_data = stock_data.iloc[:-days_to_predict]
            future_actual = stock_data.iloc[-days_to_predict:][
                [self.feature]
            ].values.astype(float)
            train_dates = training_data.index
            future_dates = stock_data.iloc[-days_to_predict:].index
        else:
            training_data = stock_data
            future_actual = None
            train_dates = training_data.index
            last_date = train_dates[-1]
            future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[
                1:
            ]

        full_data = training_data[[self.feature]].values.astype(float)
        X, y_values = self.prepare_data(training_data)
        self.train(X, y_values)

        train_predictions = self.model.predict(X, verbose=0)
        train_predictions_rescaled = self.scaler.inverse_transform(train_predictions)
        padded_train_predictions = np.full((len(full_data), 1), np.nan)
        padded_train_predictions[
            self.time_step : self.time_step + len(train_predictions)
        ] = train_predictions_rescaled

        last_n_days = training_data[[self.feature]].values[-self.time_step :]
        last_n_days_scaled = self.scaler.transform(last_n_days.astype(float))
        X_input = last_n_days_scaled.reshape(1, self.time_step, 1)
        future_predictions = self.predict(X_input, days_to_predict)

        final_predictions = (
            self.adjust_predictions(future_predictions)
            if self.use_sentiment_analysis
            else future_predictions
        )

        plot_buffer = self.plot_full_timeline(
            train_data=full_data,
            train_dates=train_dates,
            train_pred=padded_train_predictions,
            future_pred=final_predictions,
            ticker=ticker,
            future_dates=future_dates,
            future_actual=future_actual,
        )
        return plot_buffer


def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        if isinstance(ticker, str):
            ticker = ticker.upper()

    json_path = Path.home() / "Documents" / "StarZ Finance" / "ModelParameters.json"
    with open(json_path, "r") as file:
        data = json.load(file)

    lstm_config = jsonm.parse_model_parameters(data, "GRU")

    predictor = StockPredictorGRU(
        time_step=lstm_config["TimeStep"],
        feature=lstm_config["Feature"],
        gru_units=lstm_config["GRUUnits"],
        dropout_rate=lstm_config["DropoutRate"],
        epochs=lstm_config["Epochs"],
        batch_size=lstm_config["BatchSize"],
        optimizer=lstm_config["Optimizer"],
        use_early_stopping=lstm_config["UseEarlyStopping"],
        show_future_actual=lstm_config["ShowFutureActual"],
        use_sentiment_analysis=lstm_config["UseSentimentAnalysis"],
        scaling_factor=lstm_config["ScalingFactor"],
        plot_window=lstm_config["PlotWindow"],
    )

    graph_image = predictor.run(
        ticker=ticker,
        start_date=lstm_config["StartDate"],
        end_date=lstm_config["EndDate"],
        days_to_predict=lstm_config["DaysToPredict"],
    )

    graph_image.seek(0)
    base64_image = base64.b64encode(graph_image.getvalue()).decode("utf-8")
    print(base64_image)


if __name__ == "__main__":
    main()
