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
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Custom modules.
import YFinanceFetcher as yff
import JSONManager as jsonm

# Custom matplotlib style.
plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


class StockPredictorLSTM:
    def __init__(
        self,
        time_step: int = 60,
        feature: str = "Close",
        lstm_units: int = 50,
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
        """
        Initialize the LSTM stock predictor with user-provided hyperparameters.

        Parameters:
        - time_step: Number of past days to consider per training sample.
        - feature: The stock price feature to use for prediction (e.g., "Close", "Open", "High", "Low").
        - lstm_units: Number of units/neurons in each LSTM layer.
        - dropout_rate: Fraction of the neurons to drop for regularization.
        - epochs: Number of times the training algorithm works through the entire training dataset.
        - batch_size: Number of samples per gradient update during training.
        - optimizer: Optimizer to use while compiling the model.
        - use_early_stopping: Whether to use early stopping to prevent overfitting.
        - show_future_actual: Whether to split the data into training/future segments for comparison.
        - use_sentiment_analysis: If true, predictions will be adjusted by a sentiment score.
        - scaling_factor: Factor for adjusting predictions based on sentiment analysis.
        - plot_window: Number of most recent days to show in the plot, 0 shows all data.
        """
        # Hyperparameter assignments
        self.time_step = time_step
        self.feature = feature
        self.lstm_units = lstm_units
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
        """
        Prepare the dataset for training by normalizing the data and forming sliding windows.

        Parameters:
        - data: DataFrame containing the historical stock data.

        Returns:
        - Tuple containing numpy arrays for the input sequences X and target values y.
        """
        # Extract feature values and ensure they are in float format
        data_values = data[self.feature].values.reshape(-1, 1).astype(float)
        # Create scaler to transform the data to [0, 1] range
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data_values)

        X, y = [], []
        # Loop to create the sliding windows for the time series data
        for i in range(self.time_step, len(data_scaled)):
            X.append(data_scaled[i - self.time_step : i])
            y.append(data_scaled[i])
        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build and compile the LSTM model architecture.

        The model consists of:
         - An input layer accepting time series sequences.
         - Two LSTM layers (the first returns sequences for stacking).
         - Dropout layers to reduce overfitting.
         - A Dense layer to produce output predictions.

        Parameters:
        - input_shape: Shape of one training sample (number of time steps, number of features).

        Returns:
        - A compiled Keras Sequential model.
        """
        model = Sequential(
            [
                # Define the input layer with the provided shape.
                Input(shape=input_shape),
                # First LSTM layer returns full sequence for the subsequent LSTM layer.
                LSTM(units=self.lstm_units, return_sequences=True),
                # Dropout for regularization.
                Dropout(self.dropout_rate),
                # Second LSTM layer, now returning only the last output in the sequence.
                LSTM(units=self.lstm_units, return_sequences=False),
                Dropout(self.dropout_rate),
                # Dense layer that outputs the prediction.
                Dense(units=1),
            ]
        )
        # Compile the model with mean squared error loss and the chosen optimizer.
        model.compile(optimizer=self.optimizer, loss="mse")
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the LSTM model on the preprocessed training data.

        The training process includes:
         - Building the model with the input shape based on training data.
         - Optionally using early stopping to monitor the validation loss.
         - Fitting the model on the training samples with a validation split.

        Parameters:
        - X_train: Input sequences for training.
        - y_train: Actual target values for training.
        """
        # Build the model using the shape of training data.
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        # Prepare callbacks if early stopping is desired.
        if self.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=3,  # Stop if validation loss doesn't improve for 3 consecutive epochs.
                restore_best_weights=True,
            )
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0,  # Suppress training output
            )
        else:
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0,  # Suppress training output
            )

    def predict(self, X_input: np.ndarray, days_to_predict: int = 5) -> np.ndarray:
        """
        Forecast future stock values using the trained model.

        The method uses recursive prediction. For each predicted day:
         - It predicts the next value.
         - Shifts the input window to include the newly predicted value.

        Parameters:
        - X_input: The last window of data to start the forecast. Shape (1, time_step, features).
        - days_to_predict: Number of future days to forecast.

        Returns:
        - Future predicted values, transformed back to the original scale.
        """
        predicted_values = []
        for _ in range(days_to_predict):
            # Predict the next value (verbose=0 hides training messages).
            predicted = self.model.predict(X_input, verbose=0)
            # Append the predicted value (first element since predict returns a batch).
            predicted_values.append(predicted[0])
            # Update input window: remove the first time step and add the new prediction.
            X_input = np.concatenate(
                [X_input[:, 1:, :], predicted.reshape(1, 1, -1)], axis=1
            )
        predicted_values = np.array(predicted_values)
        return self.scaler.inverse_transform(predicted_values)

    def adjust_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Optionally adjust the predictions based on sentiment analysis.

        Parameters:
        - predictions: The array containing future predictions.

        Returns:
        - Adjusted prediction values.
        """
        sentiment_score = 0.5  # Placeholder for a dynamic sentiment score.
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
        """
        Create a plot showing the timeline of stock data and predictions.
        If plot_window is set, only shows the most recent portion of the training data.
        """
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
            color="#18c0c4",
            linewidth=2,
        )

        valid_train_pred = train_pred[~np.isnan(train_pred).any(axis=1)]
        valid_pred_dates = train_dates[-len(valid_train_pred) :]
        plt.plot(
            valid_pred_dates,
            valid_train_pred.reshape(-1),
            label="Training Predictions",
            color="#f3907e",
            linestyle="--",
            linewidth=2,
        )

        plt.plot(
            future_dates,
            future_pred.reshape(-1),
            label="Future Predictions",
            color="#f62196",
            linewidth=2,
        )

        if future_actual is not None:
            plt.plot(
                future_dates,
                future_actual.reshape(-1),
                label="Real Future Prices",
                color="#fefeff",
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
        plt.gcf().autofmt_xdate()  # Auto rotate date labels
        plt.tight_layout()

        buf = io.BytesIO()
        canvas = FigureCanvas(plt.gcf())
        canvas.print_png(buf)
        plt.close()  # Free up resources
        buf.seek(0)
        return buf

    def run(
        self, ticker: str, start_date: str, end_date: str, days_to_predict: int = 5
    ) -> io.BytesIO:
        """
        Execute the entire workflow: fetch data, prepare, train, predict, and plot.

        Parameters:
        - ticker: Stock ticker symbol.
        - start_date: Start date for historical data in "YYYY-MM-DD" format.
        - end_date: End date for historical data in "YYYY-MM-DD" format.
        - days_to_predict: Number of days in the future to forecast.

        Returns:
        - BytesIO object containing the plot image.
        """
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
    # Get the ticker from command line arguments.
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        if isinstance(ticker, str):
            ticker = ticker.upper()

    # ticker = "AAPL"  # Default ticker. Uncomment this line for testing.

    # Load LSTM parameters from the JSON file.
    json_path = Path.home() / "Documents" / "StarZ Finance" / "ModelParameters.json"
    with open(json_path, "r") as file:
        data = json.load(file)

    lstm_config = jsonm.parse_lstm_parameters(data)

    # Create an instance of the StockPredictorLSTM with customized parameters.
    predictor = StockPredictorLSTM(
        time_step=lstm_config["TimeStep"],
        feature=lstm_config["Feature"],
        lstm_units=lstm_config["LSTMUnits"],
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
    print(
        base64_image
    )  # This is the base64 encoded image data that gets returned to the StarZ Finance app.


if __name__ == "__main__":
    main()
