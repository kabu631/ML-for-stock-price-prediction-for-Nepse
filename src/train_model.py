#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Training Script for NEPSE Stock Price Prediction

This script handles the training of LSTM models for stock price prediction
and fraud detection.
"""

import os
import numpy as np
import pandas as pd
import logging
import argparse
import glob
from preprocessing import StockDataPreprocessor
from models import StockPriceModel, FraudDetectionModel
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
PLOTS_DIR = os.path.join("models", "plots")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_processed_data(symbol):
    """
    Load processed data for a specific symbol
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test) or None if files don't exist
    """
    # Check if processed data files exist
    x_train_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_train.npy")
    y_train_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_y_train.npy")
    x_test_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_test.npy")
    y_test_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_y_test.npy")
    
    if not all(os.path.exists(path) for path in [x_train_path, y_train_path, x_test_path, y_test_path]):
        logger.error(f"Processed data files not found for {symbol}")
        return None
    
    # Load data
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    
    return X_train, y_train, X_test, y_test


def train_stock_model(symbol, model_type, epochs, batch_size):
    """
    Train a stock price prediction model
    
    Args:
        symbol (str): Stock symbol
        model_type (str): Type of model to train ('simple', 'bidirectional', 'stacked', 'hybrid', 'advanced')
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        
    Returns:
        StockPriceModel: Trained model
    """
    logger.info(f"Training {model_type} LSTM model for {symbol}")
    
    # Load processed data
    data = load_processed_data(symbol)
    if data is None:
        # Process the data if it doesn't exist
        logger.info(f"Processed data not found for {symbol}, processing now...")
        preprocessor = StockDataPreprocessor()
        data = preprocessor.process_stock(symbol)
        
        if data is None:
            logger.error(f"Failed to process data for {symbol}")
            return None
    
    X_train, y_train, X_test, y_test = data
    
    # Create model name
    model_name = f"{symbol}_{model_type}_lstm"
    
    # Get sequence length and number of features from data shape
    sequence_length, n_features = X_train.shape[1], X_train.shape[2]
    
    # Initialize model
    model = StockPriceModel(sequence_length, n_features, name=model_name)
    
    # Build model based on type
    if model_type == 'simple':
        model.build_simple_lstm_model()
    elif model_type == 'bidirectional':
        model.build_bidirectional_lstm_model()
    elif model_type == 'stacked':
        model.build_stacked_lstm_model()
    elif model_type == 'hybrid':
        model.build_lstm_gru_hybrid_model()
    elif model_type == 'advanced':
        model.build_advanced_lstm_model()
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    # Create validation set (20% of training data)
    val_split = int(len(X_train) * 0.8)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    # Train model
    history = model.train_model(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        epochs=epochs, 
        batch_size=batch_size
    )
    
    # Evaluate model
    loss = model.evaluate_model(X_test, y_test)
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Make predictions and plot results
    plot_predictions(model, X_test, y_test, symbol, model_type)
    
    return model


def train_fraud_detection_model(symbol, epochs, batch_size):
    """
    Train a fraud detection model
    
    Args:
        symbol (str): Stock symbol
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        
    Returns:
        FraudDetectionModel: Trained model
    """
    logger.info(f"Training fraud detection model for {symbol}")
    
    # Load processed data
    data = load_processed_data(symbol)
    if data is None:
        logger.error(f"Processed data not found for {symbol}")
        return None
    
    X_train, _, X_test, _ = data
    
    # Create model name
    model_name = f"{symbol}_fraud_detection"
    
    # Get sequence length and number of features from data shape
    sequence_length, n_features = X_train.shape[1], X_train.shape[2]
    
    # Initialize and build model
    model = FraudDetectionModel(sequence_length, n_features, name=model_name)
    model.build_autoencoder_model()
    
    # Train model
    history = model.train_model(
        X_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.2
    )
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Detect anomalies
    anomaly_scores, is_anomaly = model.detect_anomalies(X_test)
    
    # Plot anomaly detection results
    plot_anomalies(anomaly_scores, is_anomaly, symbol, model.threshold)
    
    return model


def plot_training_history(history, model_name):
    """
    Plot training history
    
    Args:
        history (tensorflow.keras.callbacks.History): Training history
        model_name (str): Name of the model
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{model_name}_training_history.png")
    plt.savefig(plot_path)
    logger.info(f"Training history plot saved to {plot_path}")
    
    plt.close()


def plot_predictions(model, X_test, y_test, symbol, model_type):
    """
    Plot model predictions vs actual values
    
    Args:
        model (StockPriceModel): Trained model
        X_test (numpy.ndarray): Test sequences
        y_test (numpy.ndarray): Test targets
        symbol (str): Stock symbol
        model_type (str): Type of model
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Load target scaler for inverse transformation
    target_scaler_path = os.path.join(MODELS_DIR, "target_scaler.pkl")
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)
        
        # Inverse transform predictions and actual values
        y_test_inv = target_scaler.inverse_transform(y_test)
        y_pred_inv = target_scaler.inverse_transform(y_pred)
    else:
        # If scaler not found, just use the scaled values
        logger.warning("Target scaler not found, using scaled values for plotting")
        y_test_inv = y_test
        y_pred_inv = y_pred
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title(f'{symbol} Stock Price Prediction ({model_type} LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{symbol}_{model_type}_predictions.png")
    plt.savefig(plot_path)
    logger.info(f"Predictions plot saved to {plot_path}")
    
    plt.close()


def plot_anomalies(anomaly_scores, is_anomaly, symbol, threshold):
    """
    Plot anomaly detection results
    
    Args:
        anomaly_scores (numpy.ndarray): Reconstruction errors
        is_anomaly (numpy.ndarray): Boolean array indicating anomalies
        symbol (str): Stock symbol
        threshold (float): Anomaly threshold
    """
    plt.figure(figsize=(12, 6))
    plt.plot(anomaly_scores, label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='-', label=f'Threshold ({threshold:.4f})')
    
    # Plot anomaly points
    anomaly_indices = np.where(is_anomaly)[0]
    plt.scatter(anomaly_indices, anomaly_scores[anomaly_indices], color='red', label='Anomalies')
    
    plt.title(f'{symbol} Anomaly Detection')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{symbol}_anomalies.png")
    plt.savefig(plot_path)
    logger.info(f"Anomaly detection plot saved to {plot_path}")
    
    plt.close()


def main():
    """Main function to train models"""
    parser = argparse.ArgumentParser(description='Train LSTM models for NEPSE stock prediction')
    parser.add_argument('--symbol', type=str, help='Stock symbol to train model for (default: all)')
    parser.add_argument('--model-type', type=str, default='advanced', 
                        choices=['simple', 'bidirectional', 'stacked', 'hybrid', 'advanced'],
                        help='Type of LSTM model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--fraud-detection', action='store_true', help='Train fraud detection model')
    
    args = parser.parse_args()
    
    # Get list of available symbols
    if args.symbol:
        symbols = [args.symbol]
    else:
        # Get all symbols with processed data
        data_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_X_train.npy"))
        symbols = [os.path.basename(file).split('_X_train.npy')[0] for file in data_files]
        
        if not symbols:
            logger.error("No processed data found. Run preprocessing.py first.")
            return
    
    logger.info(f"Training models for symbols: {symbols}")
    
    for symbol in symbols:
        # Train stock price prediction model
        model = train_stock_model(symbol, args.model_type, args.epochs, args.batch_size)
        
        # Train fraud detection model if requested
        if args.fraud_detection:
            fraud_model = train_fraud_detection_model(symbol, args.epochs, args.batch_size)
    
    logger.info("Model training completed")


if __name__ == "__main__":
    main() 