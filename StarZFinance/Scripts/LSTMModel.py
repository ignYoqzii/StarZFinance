import io
import os
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Suppress TensorFlow logging messages (only show errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable TensorFlow warnings

# Import scikit-learn scaling tool for data normalization.
from sklearn.preprocessing import MinMaxScaler

# Import Keras components for building an LSTM model.
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Import Python Imaging Library to handle image operations.
from PIL import Image

# Import custom module for fetching stock market data.
import YFinanceFetcher as yff


class StockPredictorLSTM:
    def __init__(
        self,
        time_step: int = 60,
        features: list = ["Close"],
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "adam",
        use_early_stopping: bool = False,
        show_future_actual: bool = False,
        use_sentiment_analysis: bool = False,
        scaling_factor: float = 5,
    ):
        """
        Initialize the LSTM stock predictor with user-provided hyperparameters.

        Parameters:
        - time_step: Number of past days to consider per training sample.
        - features: List of feature names (columns) in the dataset to be used.
        - lstm_units: Number of units/neurons in each LSTM layer.
        - dropout_rate: Fraction of the neurons to drop for regularization.
        - epochs: Number of times the training algorithm works through the entire training dataset.
        - batch_size: Number of samples per gradient update during training.
        - optimizer: Optimizer to use while compiling the model.
        - use_early_stopping: Whether to use early stopping to prevent overfitting.
        - show_future_actual: Whether to split the data into training/future segments for comparison.
        - use_sentiment_analysis: If true, predictions will be adjusted by a sentiment score.
        - scaling_factor: Factor for adjusting predictions based on sentiment analysis.
        """
        # Hyperparameter assignments
        self.time_step = time_step
        self.features = features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.use_early_stopping = use_early_stopping
        self.show_future_actual = show_future_actual
        self.use_sentiment_analysis = use_sentiment_analysis
        self.scaling_factor = scaling_factor

        # Variables to hold the built model and the scaler after training
        self.model = None
        self.scaler = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the dataset for training by normalizing the data and forming sliding windows.

        The method performs the following:
         - Extracts the relevant features and converts them to a floating-point array.
         - Applies MinMax scaling to normalize the values between 0 and 1.
         - Creates input sequences (X) and targets (y) using a sliding window approach.

        Parameters:
        - data: DataFrame containing the historical stock data.

        Returns:
        - Tuple containing numpy arrays for the input sequences X and target values y.
        """
        # Extract required columns and ensure they are in float format.
        data_values = data[self.features].values.astype(float)
        # Create scaler to transform the data to [0, 1] range.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data_values)

        X, y = [], []
        # Loop to create the sliding windows for the time series data.
        for i in range(self.time_step, len(data_scaled)):
            # The input sequence is the data of previous "time_step" days.
            X.append(data_scaled[i - self.time_step : i])
            # The target is the data of the current day.
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
        model = Sequential([
            # Define the input layer with the provided shape.
            Input(shape=input_shape),
            # First LSTM layer returns full sequence for the subsequent LSTM layer.
            LSTM(units=self.lstm_units, return_sequences=True),
            # Dropout for regularization, randomly dropping neurons during training.
            Dropout(self.dropout_rate),
            # Second LSTM layer, now returning only the last output in the sequence.
            LSTM(units=self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            # Dense layer that outputs the prediction, dimension equals number of features.
            Dense(units=len(self.features))
        ])
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
                restore_best_weights=True
            )
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping]
            )
        else:
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2
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
        predicted_values = []  # To store predictions for each future day.
        for _ in range(days_to_predict):
            # Use the model to predict the next day. Note that verbose=0 hides training messages.
            predicted = self.model.predict(X_input, verbose=0)
            # Append the predicted value (first element since predict returns a batch).
            predicted_values.append(predicted[0])
            # Update input window: remove the first time step and add the new prediction.
            X_input = np.concatenate(
                [X_input[:, 1:, :], predicted.reshape(1, 1, -1)],
                axis=1
            )
        # Convert list of predictions to a numpy array and scale back to original values.
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
        future_actual: Optional[np.ndarray] = None
    ) -> io.BytesIO:
        """
        Create a plot showing the full timeline of stock data and predictions.

        This method plots:
         - The historical (training) stock prices.
         - The predicted stock values on the training set.
         - Future predictions.
         - Optionally, the actual future stock prices for comparison.

        Parameters:
        - train_data: The original training stock price data.
        - train_dates: Timestamps associated with the training data.
        - train_pred: Training predictions aligned to corresponding dates.
        - future_pred: Forecasted future stock prices.
        - ticker: Stock ticker symbol for labeling the plot.
        - future_dates: Future dates corresponding to the forecast.
        - future_actual: Optionally, real future stock prices (if available).

        Returns:
        - A BytesIO buffer containing the PNG image of the plot.
        """
        # Create a new figure and axes.
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the historical stock prices.
        ax.plot(train_dates, train_data.reshape(-1), label="Real Stock Prices",
                color="navy", linewidth=2)

        # Remove any rows with NaN values from training predictions for plotting.
        valid_train_pred = train_pred[~np.isnan(train_pred).any(axis=1)]
        # Adjust dates to match the predictions (skip initial time_step days).
        valid_pred_dates = train_dates[self.time_step:self.time_step + len(valid_train_pred)]
        ax.plot(valid_pred_dates, valid_train_pred.reshape(-1), label="Training Predictions",
                color="darkorange", linestyle="--", linewidth=2)

        # Plot future predictions with markers.
        ax.plot(future_dates, future_pred.reshape(-1), label="Future Predictions",
                color="forestgreen", marker="o", markersize=6)

        # If actual future values are available, plot them for comparison.
        if future_actual is not None:
            ax.plot(future_dates, future_actual.reshape(-1), label="Real Future Prices",
                    color="purple", linestyle="--")

        # Set chart title, labels, legend and grid.
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Stock Price", fontsize=12)
        ax.set_title(f"{ticker} - Full Timeline Prediction", fontsize=14)
        ax.legend()
        ax.grid(True)

        # Format x-axis dates for clarity.
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()  # Auto rotate date labels.
        fig.tight_layout()

        # Save the figure to a buffer in PNG format.
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        plt.close(fig)  # Close the figure to free up resources.
        buf.seek(0)
        return buf

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        days_to_predict: int = 5
    ) -> Tuple[np.ndarray, io.BytesIO]:
        """
        Execute the entire workflow: fetch data, prepare, train, predict, and plot.

        Steps:
         1. Fetch historical stock data using the custom YFinanceFetcher module.
         2. Check if the dataset is large enough for training and forecasting.
         3. Split the data into training and optional future actuals.
         4. Prepare data by creating sliding window samples.
         5. Train the model.
         6. Generate predictions for both the training set and future dates.
         7. Optionally adjust future predictions based on sentiment.
         8. Generate a comprehensive timeline plot.

        Parameters:
        - ticker: Stock ticker symbol.
        - start_date: Start date for historical data in "YYYY-MM-DD" format.
        - end_date: End date for historical data in "YYYY-MM-DD" format.
        - days_to_predict: Number of days in the future to forecast.

        Returns:
        - Tuple containing final predictions and a BytesIO object for the plot image.
        """
        # Fetch historical stock data for the given ticker and date range.
        stock_data = yff.fetch_stock_data(ticker, start_date, end_date)

        # Check that there is enough data for the requested time steps and forecasting.
        if len(stock_data) < self.time_step + days_to_predict:
            raise ValueError(f"Not enough data. Need at least {self.time_step + days_to_predict} days.")

        # If actual future data is to be shown, split the dataset accordingly.
        if self.show_future_actual:
            training_data = stock_data.iloc[:-days_to_predict]
            future_actual = stock_data.iloc[-days_to_predict:][self.features].values.astype(float)
            train_dates = training_data.index
            future_dates = stock_data.iloc[-days_to_predict:].index
        else:
            # Otherwise, use all available data for training and generate future dates.
            training_data = stock_data
            future_actual = None
            train_dates = training_data.index
            last_date = train_dates[-1]
            # Generate future dates starting from the last date in the dataset.
            future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[1:]

        # Extract only the chosen feature data from the training dataset.
        full_data = training_data[self.features].values.astype(float)

        # Prepare sliding window training samples and corresponding targets.
        X, y_values = self.prepare_data(training_data)
        # Train the LSTM model with prepared data.
        self.train(X, y_values)

        # Obtain predictions for the entire training set (for plotting purposes).
        train_predictions = self.model.predict(X, verbose=0)
        train_predictions_rescaled = self.scaler.inverse_transform(train_predictions)
        # Create a padded array to align predictions with original data timeline.
        padded_train_predictions = np.full((len(full_data), len(self.features)), np.nan)
        padded_train_predictions[self.time_step:self.time_step + len(train_predictions)] = train_predictions_rescaled

        # Prepare the last window of data to serve as the starting input for forecasting.
        last_n_days = training_data[self.features].values[-self.time_step:]
        last_n_days_scaled = self.scaler.transform(last_n_days.astype(float))
        X_input = last_n_days_scaled.reshape(1, self.time_step, len(self.features))
        # Predict future values.
        future_predictions = self.predict(X_input, days_to_predict)

        # Adjust future predictions if sentiment analysis is enabled.
        final_predictions = (self.adjust_predictions(future_predictions)
                             if self.use_sentiment_analysis else future_predictions)

        # Generate a timeline plot including training data and future predictions.
        plot_buffer = self.plot_full_timeline(
            train_data=full_data,
            train_dates=train_dates,
            train_pred=padded_train_predictions,
            future_pred=final_predictions,
            ticker=ticker,
            future_dates=future_dates,
            future_actual=future_actual
        )
        return final_predictions, plot_buffer


# === Test-run of the predictor ===

# Create an instance of the StockPredictorLSTM with customized parameters.
predictor = StockPredictorLSTM(
    time_step=30,
    features=["Close"],
    epochs=20,
    use_sentiment_analysis=False,
    show_future_actual=True  # When True, the last days of data are used as actual future values for comparison.
)

# Run the predictor with the specified ticker and date range.
# Ensure the range covers enough days for training and future comparison.
results, graph_image = predictor.run("AAPL", "2019-01-01", "2025-04-09", days_to_predict=31)

# Display the resulting plot from the BytesIO buffer.
graph_image.seek(0)
img = Image.open(graph_image)
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis("off")  # Hide axes for a cleaner look.
plt.show()
