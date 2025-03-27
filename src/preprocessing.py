#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preprocessing Module for NEPSE Stock Data

This module provides functionality for preprocessing, cleaning, and
feature engineering for NEPSE stock data to prepare it for model training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib
import ta
from datetime import datetime, timedelta
import glob


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class StockDataPreprocessor:
    """Class for preprocessing stock data"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = None
    
    def load_data(self, symbol=None):
        """
        Load raw stock data
        
        Args:
            symbol (str): Symbol of the specific stock to load, if None loads all
            
        Returns:
            pandas.DataFrame or dict: DataFrame for specific symbol or dict of DataFrames
        """
        if symbol:
            file_path = os.path.join(RAW_DATA_DIR, f"{symbol}_historical.csv")
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            logger.info(f"Loading data for {symbol}")
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                return None
        else:
            # Load all stock data files
            data_files = glob.glob(os.path.join(RAW_DATA_DIR, "*_historical.csv"))
            data = {}
            
            for file_path in data_files:
                symbol = os.path.basename(file_path).replace("_historical.csv", "")
                try:
                    df = pd.read_csv(file_path)
                    data[symbol] = df
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {str(e)}")
            
            return data
    
    def clean_data(self, df):
        """
        Clean and preprocess raw data
        
        Args:
            df (pandas.DataFrame): Raw stock data
            
        Returns:
            pandas.DataFrame: Cleaned data
        """
        if df is None or df.empty:
            logger.error("No data to clean")
            return None
            
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Handle missing values
        if df.isnull().any().any():
            logger.info("Handling missing values")
            
            # Forward fill for most columns
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(method='ffill')
            
            # For volume, fill with 0 or median
            if 'volume' in df.columns and df['volume'].isnull().any():
                df['volume'] = df['volume'].fillna(df['volume'].median())
        
        # Check for duplicate dates
        if df.duplicated('date').any():
            logger.info("Removing duplicate dates")
            df = df.drop_duplicates('date', keep='first')
        
        return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the data
        
        Args:
            df (pandas.DataFrame): Clean stock data
            
        Returns:
            pandas.DataFrame: Data with technical indicators
        """
        if df is None or df.empty:
            logger.error("No data for adding technical indicators")
            return None
            
        logger.info("Adding technical indicators")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns for technical indicators. Required: {required_cols}")
            return df
        
        try:
            # Add simple moving averages
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # Add exponential moving averages
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Add MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Add RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # Add Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bollinger_mavg'] = bollinger.bollinger_mavg()
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_low'] = bollinger.bollinger_lband()
            
            # Add volume indicators
            df['volume_ema'] = ta.trend.ema_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_ema']
            
            # Calculate returns
            df['daily_return'] = df['close'].pct_change()
            df['5d_return'] = df['close'].pct_change(periods=5)
            df['10d_return'] = df['close'].pct_change(periods=10)
            
            # Volatility (20-day rolling standard deviation of returns)
            df['volatility'] = df['daily_return'].rolling(window=20).std()
            
            # Simple momentum indicators
            df['momentum_5d'] = df['close'] - df['close'].shift(5)
            df['momentum_10d'] = df['close'] - df['close'].shift(10)
            
            # Price change rate
            df['price_change_rate'] = (df['close'] - df['open']) / df['open']
            
            # High-Low difference
            df['high_low_diff'] = (df['high'] - df['low']) / df['low']
            
            # Fill NaN values that result from rolling operations
            df = df.fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def create_sequences(self, df, sequence_length=60, target_column='close', features=None):
        """
        Create sequences for LSTM model
        
        Args:
            df (pandas.DataFrame): Processed stock data
            sequence_length (int): Length of input sequences
            target_column (str): Column to predict
            features (list): List of feature columns to use
            
        Returns:
            tuple: (X, y) where X are the input sequences and y are the target values
        """
        if df is None or df.empty:
            logger.error("No data for creating sequences")
            return None, None
            
        if features is None:
            # Use all numeric columns except date and target as features
            features = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'date' and col != target_column]
        
        # Ensure all features are in the dataframe
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            logger.error(f"Missing features in dataframe: {missing_features}")
            return None, None
        
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Extract features
        data = df[features].values
        
        # Scale features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data)
        
        # Create separate scaler for the target column to make it easier to inverse transform later
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_values = df[target_column].values.reshape(-1, 1)
        target_scaler.fit(target_values)
        
        # Save scalers
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, "feature_scaler.pkl"))
        joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler.pkl"))
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i+sequence_length])
            y.append(target_values[i+sequence_length])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale target values
        y = target_scaler.transform(y)
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.8):
        """
        Split data into training and testing sets
        
        Args:
            X (numpy.ndarray): Input sequences
            y (numpy.ndarray): Target values
            train_ratio (float): Ratio of training data
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if X is None or y is None:
            logger.error("No data to split")
            return None, None, None, None
            
        logger.info(f"Splitting data with train ratio {train_ratio}")
        
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def save_processed_data(self, symbol, data):
        """
        Save processed data to file
        
        Args:
            symbol (str): Stock symbol
            data (pandas.DataFrame): Processed data
            
        Returns:
            str: Path to saved file
        """
        if data is None or data.empty:
            logger.error("No data to save")
            return None
            
        output_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_processed.csv")
        data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path
    
    def process_stock(self, symbol):
        """
        Process a single stock's data end-to-end
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test) or None if processing fails
        """
        logger.info(f"Processing stock: {symbol}")
        
        # Load data
        df = self.load_data(symbol)
        if df is None:
            return None
        
        # Clean data
        df = self.clean_data(df)
        if df is None:
            return None
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        if df is None:
            return None
        
        # Save processed data
        self.save_processed_data(symbol, df)
        
        # Create sequences
        X, y = self.create_sequences(df)
        if X is None or y is None:
            return None
        
        # Split data
        X_train, y_train, X_test, y_test = self.split_data(X, y)
        if X_train is None:
            return None
        
        # Save the processed sequences
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{symbol}_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_test.npy"), X_test)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{symbol}_y_test.npy"), y_test)
        
        logger.info(f"Successfully processed {symbol}")
        return X_train, y_train, X_test, y_test


def main():
    """Main function to preprocess all stock data"""
    logger.info("Starting data preprocessing")
    preprocessor = StockDataPreprocessor()
    
    # Get list of all stock data files
    data_files = glob.glob(os.path.join(RAW_DATA_DIR, "*_historical.csv"))
    
    if not data_files:
        logger.error("No stock data files found")
        return
    
    for file_path in data_files:
        symbol = os.path.basename(file_path).replace("_historical.csv", "")
        logger.info(f"Processing {symbol}")
        preprocessor.process_stock(symbol)
    
    logger.info("Data preprocessing completed")


if __name__ == "__main__":
    main() 