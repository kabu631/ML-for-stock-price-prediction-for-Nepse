#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM Models for NEPSE Stock Price Prediction

This module contains the neural network model definitions used for
stock price prediction and fraud detection.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = os.path.join("models")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class StockPriceModel:
    """LSTM models for stock price prediction"""
    
    def __init__(self, sequence_length, n_features, name="lstm_model"):
        """
        Initialize Stock Price Model
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features per time step
            name (str): Name of the model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.name = name
        self.model = None
        
    def build_simple_lstm_model(self):
        """
        Build a simple LSTM model
        
        Returns:
            tensorflow.keras.models.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Simple LSTM model built")
        
        self.model = model
        return model
    
    def build_bidirectional_lstm_model(self):
        """
        Build a bidirectional LSTM model
        
        Returns:
            tensorflow.keras.models.Model: Compiled Bidirectional LSTM model
        """
        model = Sequential([
            Bidirectional(LSTM(units=50, return_sequences=True), 
                          input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            Bidirectional(LSTM(units=50, return_sequences=False)),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Bidirectional LSTM model built")
        
        self.model = model
        return model
    
    def build_stacked_lstm_model(self):
        """
        Build a stacked LSTM model
        
        Returns:
            tensorflow.keras.models.Model: Compiled Stacked LSTM model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Stacked LSTM model built")
        
        self.model = model
        return model
    
    def build_lstm_gru_hybrid_model(self):
        """
        Build a hybrid LSTM-GRU model
        
        Returns:
            tensorflow.keras.models.Model: Compiled hybrid model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM-GRU hybrid model built")
        
        self.model = model
        return model
    
    def build_advanced_lstm_model(self):
        """
        Build an advanced LSTM model with tuned hyperparameters
        
        Returns:
            tensorflow.keras.models.Model: Compiled advanced LSTM model
        """
        model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True), 
                          input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.3),
            Dense(units=32, activation='relu'),
            Dropout(0.2),
            Dense(units=16, activation='relu'),
            Dense(units=1)
        ])
        
        # Use Adam optimizer with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        logger.info("Advanced LSTM model built")
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train (numpy.ndarray): Training sequences
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray): Validation sequences
            y_val (numpy.ndarray): Validation targets
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            logger.error("Model not built yet. Call one of the build_*_model methods first.")
            return None
        
        logger.info(f"Training model with {epochs} epochs and batch size {batch_size}")
        
        # Create model checkpoint callback
        checkpoint_path = os.path.join(MODELS_DIR, f"{self.name}_checkpoint.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        
        # Create learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        # Save the final model
        model_path = os.path.join(MODELS_DIR, f"{self.name}.h5")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test (numpy.ndarray): Test sequences
            y_test (numpy.ndarray): Test targets
            
        Returns:
            float: Test loss
        """
        if self.model is None:
            logger.error("Model not built yet. Call one of the build_*_model methods first.")
            return None
        
        logger.info("Evaluating model on test data")
        
        loss = self.model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test loss: {loss}")
        
        return loss
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X (numpy.ndarray): Input sequences
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.model is None:
            logger.error("Model not built yet. Call one of the build_*_model methods first.")
            return None
        
        logger.info(f"Making predictions with {self.name}")
        
        predictions = self.model.predict(X)
        return predictions
    
    def load_model(self, model_path=None):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to saved model, if None uses default path
            
        Returns:
            tensorflow.keras.models.Model: Loaded model
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f"{self.name}.h5")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Loading model from {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        return self.model


class FraudDetectionModel:
    """Anomaly detection model for potential fraud"""
    
    def __init__(self, sequence_length, n_features, name="fraud_detection_model"):
        """
        Initialize Fraud Detection Model
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features per time step
            name (str): Name of the model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.name = name
        self.model = None
        self.threshold = None
    
    def build_autoencoder_model(self):
        """
        Build an autoencoder model for anomaly detection
        
        Returns:
            tensorflow.keras.models.Model: Compiled autoencoder model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder layers
        encoded = LSTM(units=32, return_sequences=True)(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(units=16, return_sequences=False)(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(units=8)(encoded)
        
        # Decoder layers (we recreate the sequence)
        decoded = Dense(units=16)(encoded)
        decoded = Dense(units=32)(decoded)
        # Reshape to match the original dimension
        decoded = Dense(units=self.sequence_length * self.n_features)(decoded)
        decoded = tf.keras.layers.Reshape((self.sequence_length, self.n_features))(decoded)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=decoded)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logger.info("Autoencoder model for fraud detection built")
        
        self.model = model
        return model
    
    def train_model(self, X_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the autoencoder model
        
        Args:
            X_train (numpy.ndarray): Training sequences
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            validation_split (float): Portion of data to use for validation
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            logger.error("Model not built yet. Call build_autoencoder_model first.")
            return None
        
        logger.info(f"Training fraud detection model with {epochs} epochs and batch size {batch_size}")
        
        # Create callbacks
        checkpoint_path = os.path.join(MODELS_DIR, f"{self.name}_checkpoint.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model (autoencoder tries to reconstruct the input)
        history = self.model.fit(
            X_train, X_train,  # Input and output are the same for autoencoders
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Save the final model
        model_path = os.path.join(MODELS_DIR, f"{self.name}.h5")
        self.model.save(model_path)
        logger.info(f"Fraud detection model saved to {model_path}")
        
        # Calculate the reconstruction error threshold
        reconstructions = self.model.predict(X_train)
        # MSE for each sample
        mse = np.mean(np.square(X_train - reconstructions), axis=(1, 2))
        # Set threshold as mean + 2*std (can be adjusted)
        self.threshold = np.mean(mse) + 2 * np.std(mse)
        
        # Save the threshold
        threshold_path = os.path.join(MODELS_DIR, f"{self.name}_threshold.npy")
        np.save(threshold_path, self.threshold)
        logger.info(f"Anomaly threshold set to {self.threshold} and saved to {threshold_path}")
        
        return history
    
    def detect_anomalies(self, X):
        """
        Detect anomalies in the data
        
        Args:
            X (numpy.ndarray): Input sequences
            
        Returns:
            tuple: (anomaly_scores, is_anomaly)
        """
        if self.model is None:
            logger.error("Model not built yet. Call build_autoencoder_model first.")
            return None, None
        
        if self.threshold is None:
            logger.error("Threshold not set. Call train_model first or load threshold.")
            return None, None
        
        logger.info("Detecting anomalies")
        
        # Get reconstructions
        reconstructions = self.model.predict(X)
        
        # Calculate mean squared error
        mse = np.mean(np.square(X - reconstructions), axis=(1, 2))
        
        # Determine which points are anomalies
        is_anomaly = mse > self.threshold
        
        return mse, is_anomaly
    
    def load_model(self, model_path=None, threshold_path=None):
        """
        Load a saved model and threshold
        
        Args:
            model_path (str): Path to saved model, if None uses default path
            threshold_path (str): Path to saved threshold, if None uses default path
            
        Returns:
            tensorflow.keras.models.Model: Loaded model
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f"{self.name}.h5")
        
        if threshold_path is None:
            threshold_path = os.path.join(MODELS_DIR, f"{self.name}_threshold.npy")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Loading fraud detection model from {model_path}")
        
        # Custom loading with proper loss function
        try:
            # First try normal loading
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.warning(f"Standard loading failed, trying custom loading: {str(e)}")
            # If that fails, create a fresh model with the same architecture and load weights
            self.build_autoencoder_model()
            self.model.load_weights(model_path)
        
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)
            logger.info(f"Loaded anomaly threshold: {self.threshold}")
        else:
            logger.warning(f"Threshold file not found: {threshold_path}")
        
        return self.model 