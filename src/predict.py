#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction Script for NEPSE Stock Price Prediction

This script handles making predictions with trained LSTM models
for future stock prices and detecting potential fraud.
"""

import os
import numpy as np
import pandas as pd
import logging
import argparse
import glob
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from preprocessing import StockDataPreprocessor
from models import StockPriceModel, FraudDetectionModel
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
PLOTS_DIR = os.path.join("models", "plots")
RAW_DATA_DIR = os.path.join("data", "raw")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model(symbol, model_type='advanced'):
    """
    Load a trained model
    
    Args:
        symbol (str): Stock symbol
        model_type (str): Type of model ('simple', 'bidirectional', 'stacked', 'hybrid', 'advanced')
        
    Returns:
        tuple: (StockPriceModel, sequence_length, n_features) or None if model not found
    """
    model_name = f"{symbol}_{model_type}_lstm"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    # Load feature scaler
    feature_scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    if not os.path.exists(feature_scaler_path):
        logger.error(f"Feature scaler not found at {feature_scaler_path}")
        return None
    
    # Get data shape from saved arrays to determine sequence_length and n_features
    x_test_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_test.npy")
    if os.path.exists(x_test_path):
        X_test = np.load(x_test_path)
        sequence_length, n_features = X_test.shape[1], X_test.shape[2]
    else:
        # Default values if test data not found
        logger.warning(f"Test data not found, using default model dimensions")
        sequence_length, n_features = 60, 30  # Default values, may need adjustment
    
    # Initialize and load model
    model = StockPriceModel(sequence_length, n_features, name=model_name)
    model.load_model(model_path)
    
    return model, sequence_length, n_features


def load_fraud_detection_model(symbol):
    """
    Load a trained fraud detection model
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        tuple: (FraudDetectionModel, sequence_length, n_features) or None if model not found
    """
    model_name = f"{symbol}_fraud_detection"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    threshold_path = os.path.join(MODELS_DIR, f"{model_name}_threshold.npy")
    
    if not os.path.exists(model_path):
        logger.error(f"Fraud detection model not found at {model_path}")
        return None
    
    # Get data shape from saved arrays
    x_test_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_test.npy")
    if os.path.exists(x_test_path):
        X_test = np.load(x_test_path)
        sequence_length, n_features = X_test.shape[1], X_test.shape[2]
    else:
        # Default values if test data not found
        logger.warning(f"Test data not found, using default model dimensions")
        sequence_length, n_features = 60, 30  # Default values, may need adjustment
    
    # Initialize and load model
    model = FraudDetectionModel(sequence_length, n_features, name=model_name)
    model.load_model(model_path, threshold_path)
    
    return model, sequence_length, n_features


def prepare_latest_data(symbol, sequence_length, preprocessor=None):
    """
    Prepare the latest data for prediction
    
    Args:
        symbol (str): Stock symbol
        sequence_length (int): Length of input sequences
        preprocessor (StockDataPreprocessor): Preprocessor instance
        
    Returns:
        numpy.ndarray: Prepared sequence for prediction
    """
    if preprocessor is None:
        preprocessor = StockDataPreprocessor()
    
    # Load and preprocess latest data
    df = preprocessor.load_data(symbol)
    if df is None:
        logger.error(f"Could not load data for {symbol}")
        return None
    
    # Clean and add features
    df = preprocessor.clean_data(df)
    df = preprocessor.add_technical_indicators(df)
    
    if df is None or df.empty:
        logger.error(f"Failed to process data for {symbol}")
        return None
    
    # Load feature scaler
    feature_scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    if not os.path.exists(feature_scaler_path):
        logger.error(f"Feature scaler not found at {feature_scaler_path}")
        return None
    
    feature_scaler = joblib.load(feature_scaler_path)
    
    # Get features (exclude date and target)
    features = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'date' and col != 'close']
    
    # Extract the latest sequence
    latest_data = df[features].values
    
    # Scale the data
    latest_data_scaled = feature_scaler.transform(latest_data)
    
    # Create a sequence of appropriate length
    if len(latest_data_scaled) < sequence_length:
        logger.error(f"Not enough data points for {symbol} (need {sequence_length}, have {len(latest_data_scaled)})")
        return None
    
    # Get the latest sequence
    latest_sequence = latest_data_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
    
    return latest_sequence


def predict_future_prices(model, latest_sequence, days=30, symbol=None, model_type=None):
    """
    Predict future stock prices
    
    Args:
        model (StockPriceModel): Trained model
        latest_sequence (numpy.ndarray): Latest data sequence
        days (int): Number of days to predict
        symbol (str): Stock symbol
        model_type (str): Type of model
        
    Returns:
        numpy.ndarray: Predicted prices
    """
    if model is None or latest_sequence is None:
        logger.error("Model or latest sequence is None")
        return None
    
    # Load target scaler for inverse transformation
    target_scaler_path = os.path.join(MODELS_DIR, "target_scaler.pkl")
    if not os.path.exists(target_scaler_path):
        logger.error(f"Target scaler not found at {target_scaler_path}")
        return None
    
    target_scaler = joblib.load(target_scaler_path)
    
    # Make iterative predictions
    curr_sequence = latest_sequence.copy()
    predictions = []
    
    for i in range(days):
        # Predict next value
        pred = model.predict(curr_sequence)
        predictions.append(pred[0][0])
        
        # Update sequence for next prediction
        # Remove oldest observation, add prediction as the newest observation
        curr_sequence = np.roll(curr_sequence, -1, axis=1)
        
        # This assumes the target (close price) is the last feature
        # Adjust this if your feature order is different
        curr_sequence[0, -1, -1] = pred[0][0]
    
    # Convert predictions from list to numpy array
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Inverse transform predictions
    predictions_inv = target_scaler.inverse_transform(predictions)
    
    # Plot future predictions if symbol is provided
    if symbol and model_type:
        plot_future_predictions(predictions_inv, symbol, model_type, days)
    
    return predictions_inv


def detect_fraud_in_latest_data(model, latest_sequence, symbol=None):
    """
    Detect fraud in the latest data
    
    Args:
        model (FraudDetectionModel): Trained fraud detection model
        latest_sequence (numpy.ndarray): Latest data sequence
        symbol (str): Stock symbol
        
    Returns:
        tuple: (anomaly_score, is_anomaly)
    """
    if model is None or latest_sequence is None:
        logger.error("Model or latest sequence is None")
        return None, None
    
    # Detect anomalies
    anomaly_scores, is_anomaly = model.detect_anomalies(latest_sequence)
    
    logger.info(f"Anomaly detection for {symbol}: score={anomaly_scores[0]}, threshold={model.threshold}, is_anomaly={is_anomaly[0]}")
    
    return anomaly_scores[0], is_anomaly[0]


def plot_future_predictions(predictions, symbol, model_type, days):
    """
    Plot future price predictions
    
    Args:
        predictions (numpy.ndarray): Predicted prices
        symbol (str): Stock symbol
        model_type (str): Type of model
        days (int): Number of days predicted
    """
    # Create date range for x-axis
    future_dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, predictions, 'b-', label='Predicted Price')
    
    plt.title(f'{symbol} Future Price Prediction ({model_type} LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{symbol}_{model_type}_future_predictions.png")
    plt.savefig(plot_path)
    logger.info(f"Future predictions plot saved to {plot_path}")
    
    plt.close()


def main():
    """Main function to make predictions"""
    parser = argparse.ArgumentParser(description='Make predictions with trained LSTM models')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to predict')
    parser.add_argument('--model-type', type=str, default='advanced', 
                        choices=['simple', 'bidirectional', 'stacked', 'hybrid', 'advanced'],
                        help='Type of model to use')
    parser.add_argument('--days', type=int, default=30, help='Number of days to predict')
    parser.add_argument('--fraud-detection', action='store_true', help='Run fraud detection')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor()
    
    # Load prediction model
    model_data = load_model(args.symbol, args.model_type)
    if model_data is None:
        logger.error(f"Could not load model for {args.symbol}")
        return
    
    model, sequence_length, n_features = model_data
    
    # Prepare latest data
    latest_sequence = prepare_latest_data(args.symbol, sequence_length, preprocessor)
    if latest_sequence is None:
        logger.error(f"Could not prepare latest data for {args.symbol}")
        return
    
    # Make predictions
    predictions = predict_future_prices(
        model, latest_sequence, args.days, args.symbol, args.model_type
    )
    
    if predictions is not None:
        logger.info(f"Future price predictions for {args.symbol} (next {args.days} days):")
        for i, price in enumerate(predictions):
            date = datetime.now() + timedelta(days=i)
            logger.info(f"  {date.strftime('%Y-%m-%d')}: {price[0]:.2f}")
    
    # Run fraud detection if requested
    if args.fraud_detection:
        # Load fraud detection model
        fraud_model_data = load_fraud_detection_model(args.symbol)
        if fraud_model_data is None:
            logger.error(f"Could not load fraud detection model for {args.symbol}")
            return
        
        fraud_model, fraud_seq_len, fraud_n_features = fraud_model_data
        
        # Prepare latest data for fraud detection
        # If sequence lengths are different, we need to prepare a different sequence
        if fraud_seq_len != sequence_length:
            fraud_sequence = prepare_latest_data(args.symbol, fraud_seq_len, preprocessor)
        else:
            fraud_sequence = latest_sequence
        
        if fraud_sequence is None:
            logger.error(f"Could not prepare latest data for fraud detection")
            return
        
        # Detect fraud
        anomaly_score, is_anomaly = detect_fraud_in_latest_data(fraud_model, fraud_sequence, args.symbol)
        
        if is_anomaly is not None:
            if is_anomaly:
                logger.warning(f"FRAUD ALERT: Potential fraud detected for {args.symbol}! Anomaly score: {anomaly_score:.4f}")
            else:
                logger.info(f"No fraud detected for {args.symbol}. Anomaly score: {anomaly_score:.4f}")


if __name__ == "__main__":
    main() 