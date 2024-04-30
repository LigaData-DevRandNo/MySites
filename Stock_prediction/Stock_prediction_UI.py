from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import csv
import base64
import os

app = Flask(__name__)

# Load the pre-trained LSTM model
# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir) 


model = load_model(os.path.join(FILE_DIR,"model.h5"))

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data
def preprocess_data(data):
    processed_data = data.dropna()
    return processed_data

# Function to scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Function to make predictions
def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

# Function to visualize predictions
def visualize_predictions(actual, predicted):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(actual, label='Actual Prices')
    axis.plot(predicted, label='Predicted Prices')
    axis.set_xlabel('Time')
    axis.set_ylabel('Price')
    axis.set_title('Actual vs. Predicted Prices')
    axis.legend()
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

# Function to read tickers from CSV
def read_tickers_from_csv(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        reader = csv.reader(file)
        next(reader)
        tickers = [(row[0],row[2]) for row in reader]
    return tickers

# Route for home page
@app.route('/')
def index():
    tickers = read_tickers_from_csv(os.path.join(FILE_DIR,'stocks.csv'))
    return render_template('index.html', tickers=tickers)
    
# Route for predicting stock prices
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected ticker from the form
    ticker_tuple = request.form['ticker']
    selected_ticker = ticker_tuple.split(',')[0].strip("('").strip("'").strip() # Extract the first element of the tuple
    selected_company = ticker_tuple.split(',')[1].strip("')").strip("'").strip()
    start_date = '2010-01-01'
    end_date = '2022-01-01'

    # Get stock data
    stock_data = get_stock_data(selected_ticker, start_date, end_date)
    processed_data = preprocess_data(stock_data)
    close_prices = processed_data['Close'].values

    # Scale data
    scaled_data, scaler = scale_data(close_prices)

    # Prepare data for prediction
    sequence_length = 60
    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Make predictions
    predictions = make_predictions(model, X)
    predictions = scaler.inverse_transform(predictions)

    # Visualize predictions
    plot_url = visualize_predictions(close_prices[sequence_length:], predictions)

    return render_template('result.html', ticker=selected_company, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)