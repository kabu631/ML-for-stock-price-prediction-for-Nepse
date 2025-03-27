#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fraud Detection Module for NEPSE Stock Market

This module provides comprehensive fraud detection analysis for NEPSE stocks,
combining the autoencoder-based anomaly detection with additional statistical
and pattern-based methods to identify potential market manipulation.
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
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from preprocessing import StockDataPreprocessor
from models import FraudDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
PLOTS_DIR = os.path.join("models", "plots")
REPORTS_DIR = os.path.join("models", "reports")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class FraudAnalyzer:
    """Class for comprehensive fraud analysis in stock data"""
    
    def __init__(self):
        """Initialize the fraud analyzer"""
        self.preprocessor = StockDataPreprocessor()
        self.fraud_indicators = []
    
    def load_processed_data(self, symbol):
        """
        Load processed data for a specific symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pandas.DataFrame: Processed data
        """
        file_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_processed.csv")
        if not os.path.exists(file_path):
            logger.error(f"Processed data file not found: {file_path}")
            return None
        
        logger.info(f"Loading processed data for {symbol}")
        try:
            df = pd.read_csv(file_path)
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Error loading processed data for {symbol}: {str(e)}")
            return None
    
    def run_autoencoder_detection(self, symbol):
        """
        Run autoencoder-based anomaly detection
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            tuple: (anomaly_scores, is_anomaly, dataframe with anomaly flags)
        """
        logger.info(f"Running autoencoder anomaly detection for {symbol}")
        
        # Load autoencoder model
        model_name = f"{symbol}_fraud_detection"
        model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
        threshold_path = os.path.join(MODELS_DIR, f"{model_name}_threshold.npy")
        
        if not os.path.exists(model_path) or not os.path.exists(threshold_path):
            logger.error(f"Autoencoder model or threshold not found for {symbol}")
            return None, None, None
        
        # Load test data
        x_test_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_X_test.npy")
        if not os.path.exists(x_test_path):
            logger.error(f"Test data not found for {symbol}")
            return None, None, None
        
        X_test = np.load(x_test_path)
        sequence_length, n_features = X_test.shape[1], X_test.shape[2]
        
        # Load full dataset to match with anomalies
        df = self.load_processed_data(symbol)
        if df is None:
            return None, None, None
        
        # Initialize and load model
        model = FraudDetectionModel(sequence_length, n_features, name=model_name)
        model.load_model(model_path, threshold_path)
        
        # Detect anomalies
        anomaly_scores, is_anomaly = model.detect_anomalies(X_test)
        
        # Add anomaly information to dataframe
        # The test data starts after sequence_length rows
        test_start_idx = sequence_length
        test_df = df.iloc[test_start_idx:test_start_idx+len(X_test)].copy()
        
        if len(test_df) == len(anomaly_scores):
            test_df['autoencoder_anomaly_score'] = anomaly_scores
            test_df['autoencoder_is_anomaly'] = is_anomaly
            self.fraud_indicators.append('autoencoder_is_anomaly')
            return anomaly_scores, is_anomaly, test_df
        else:
            logger.error(f"Length mismatch between test data and anomaly scores for {symbol}")
            return anomaly_scores, is_anomaly, None
    
    def detect_price_manipulation(self, df):
        """
        Detect potential price manipulation patterns
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Data with price manipulation flags
        """
        if df is None or df.empty:
            return None
        
        logger.info("Detecting price manipulation patterns")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # 1. Detect unusual price changes (Z-score based)
        if 'daily_return' in result_df.columns:
            # Calculate z-score of daily returns
            z_scores = stats.zscore(result_df['daily_return'].fillna(0))
            # Flag returns more than 3 standard deviations away from mean
            result_df['unusual_return'] = abs(z_scores) > 3.0
            self.fraud_indicators.append('unusual_return')
        
        # 2. Detect unusual volume spikes
        if 'volume' in result_df.columns:
            # Calculate relative volume compared to moving average
            result_df['volume_ratio'] = result_df['volume'] / result_df['volume'].rolling(20).mean()
            # Flag volume more than 5 times the average
            result_df['unusual_volume'] = result_df['volume_ratio'] > 5.0
            self.fraud_indicators.append('unusual_volume')
        
        # 3. Detect price-volume divergence (potential manipulation)
        if 'daily_return' in result_df.columns and 'volume_ratio' in result_df.columns:
            # High returns with low volume or vice versa can be suspicious
            result_df['price_volume_divergence'] = (
                (result_df['daily_return'].abs() > 0.03) & 
                (result_df['volume_ratio'] < 0.5)
            ) | (
                (result_df['daily_return'].abs() < 0.005) & 
                (result_df['volume_ratio'] > 3.0)
            )
            self.fraud_indicators.append('price_volume_divergence')
        
        # 4. Detect potential "pump and dump" patterns
        # Look for rapid price increase followed by decrease
        if 'close' in result_df.columns:
            # Calculate 5-day price changes
            result_df['5d_price_change'] = result_df['close'].pct_change(periods=5)
            result_df['5d_future_change'] = result_df['close'].shift(-5).pct_change(periods=5)
            
            # Potential pump and dump: significant rise followed by significant fall
            result_df['pump_and_dump'] = (
                (result_df['5d_price_change'] > 0.1) & 
                (result_df['5d_future_change'] < -0.1)
            )
            self.fraud_indicators.append('pump_and_dump')
        
        # Fill NaN values
        for col in self.fraud_indicators:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(False)
        
        return result_df
    
    def detect_statistical_anomalies(self, df):
        """
        Detect statistical anomalies using machine learning methods
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Data with statistical anomaly flags
        """
        if df is None or df.empty:
            return None
        
        logger.info("Detecting statistical anomalies")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Select numerical features for anomaly detection
        features = [col for col in df.select_dtypes(include=[np.number]).columns 
                  if col not in ['date'] and not col.startswith('is_') 
                  and not col.endswith('_anomaly')]
        
        # Handle missing values
        X = result_df[features].fillna(0)
        
        # 1. Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        result_df['isolation_forest_anomaly'] = iso_forest.fit_predict(X) == -1
        self.fraud_indicators.append('isolation_forest_anomaly')
        
        # 2. K-Nearest Neighbors Anomaly Detection
        knn = KNN(contamination=0.05)
        knn.fit(X)
        result_df['knn_anomaly'] = knn.predict(X) == 1
        self.fraud_indicators.append('knn_anomaly')
        
        return result_df
    
    def calculate_combined_anomaly_score(self, df):
        """
        Calculate a combined anomaly score based on all indicators
        
        Args:
            df (pandas.DataFrame): Stock data with fraud indicators
            
        Returns:
            pandas.DataFrame: Data with combined anomaly score
        """
        if df is None or df.empty or not self.fraud_indicators:
            return df
        
        logger.info("Calculating combined anomaly score")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate weighted sum of indicators (if column exists, add to sum)
        # Higher weights for more reliable indicators
        weights = {
            'autoencoder_is_anomaly': 0.3,
            'isolation_forest_anomaly': 0.2,
            'knn_anomaly': 0.2,
            'unusual_return': 0.1,
            'unusual_volume': 0.1,
            'price_volume_divergence': 0.05,
            'pump_and_dump': 0.05
        }
        
        # Initialize score column
        result_df['combined_anomaly_score'] = 0.0
        
        # Add weighted contributions
        total_weight = 0.0
        for indicator, weight in weights.items():
            if indicator in result_df.columns:
                result_df['combined_anomaly_score'] += weight * result_df[indicator].astype(float)
                total_weight += weight
        
        # Normalize by total weight applied
        if total_weight > 0:
            result_df['combined_anomaly_score'] /= total_weight
        
        # Flag significant anomalies
        result_df['is_significant_anomaly'] = result_df['combined_anomaly_score'] > 0.5
        
        return result_df
    
    def analyze_stock(self, symbol):
        """
        Run comprehensive fraud analysis for a stock
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pandas.DataFrame: Analysis results
        """
        logger.info(f"Starting comprehensive fraud analysis for {symbol}")
        
        # Reset fraud indicators
        self.fraud_indicators = []
        
        # 1. Run autoencoder detection
        anomaly_scores, is_anomaly, df = self.run_autoencoder_detection(symbol)
        
        if df is None:
            logger.error(f"Autoencoder detection failed for {symbol}")
            return None
        
        # 2. Detect price manipulation patterns
        df = self.detect_price_manipulation(df)
        
        # 3. Detect statistical anomalies
        df = self.detect_statistical_anomalies(df)
        
        # 4. Calculate combined anomaly score
        df = self.calculate_combined_anomaly_score(df)
        
        # 5. Generate visualizations
        self.visualize_anomalies(df, symbol)
        
        # 6. Save results
        self.save_analysis_results(df, symbol)
        
        return df
    
    def visualize_anomalies(self, df, symbol):
        """
        Generate visualizations for detected anomalies
        
        Args:
            df (pandas.DataFrame): Analysis results
            symbol (str): Stock symbol
        """
        if df is None or df.empty:
            return
        
        logger.info(f"Generating anomaly visualizations for {symbol}")
        
        # 1. Price chart with anomalies highlighted
        plt.figure(figsize=(14, 8))
        
        # Plot stock price
        plt.subplot(2, 1, 1)
        plt.plot(df['date'], df['close'], 'b-', label='Close Price')
        
        # Highlight anomalies
        if 'is_significant_anomaly' in df.columns:
            anomaly_points = df[df['is_significant_anomaly']]['date']
            anomaly_prices = df[df['is_significant_anomaly']]['close']
            plt.scatter(anomaly_points, anomaly_prices, color='red', s=50, label='Significant Anomalies')
        
        plt.title(f'{symbol} - Stock Price with Anomalies')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot combined anomaly score
        plt.subplot(2, 1, 2)
        plt.plot(df['date'], df['combined_anomaly_score'], 'g-', label='Anomaly Score')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        plt.title(f'{symbol} - Combined Anomaly Score')
        plt.xlabel('Date')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(PLOTS_DIR, f"{symbol}_fraud_analysis.png")
        plt.savefig(output_path)
        logger.info(f"Saved fraud analysis plot to {output_path}")
        plt.close()
        
        # 2. Heatmap of correlations between fraud indicators
        if len(self.fraud_indicators) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[self.fraud_indicators].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'{symbol} - Correlation Between Fraud Indicators')
            
            # Save figure
            output_path = os.path.join(PLOTS_DIR, f"{symbol}_fraud_indicator_correlation.png")
            plt.savefig(output_path)
            logger.info(f"Saved fraud indicator correlation plot to {output_path}")
            plt.close()
    
    def save_analysis_results(self, df, symbol):
        """
        Save fraud analysis results
        
        Args:
            df (pandas.DataFrame): Analysis results
            symbol (str): Stock symbol
        """
        if df is None or df.empty:
            return
        
        # 1. Save the detailed results DataFrame
        output_path = os.path.join(REPORTS_DIR, f"{symbol}_fraud_analysis.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed fraud analysis to {output_path}")
        
        # 2. Generate and save a summary report
        summary = {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'total_days_analyzed': len(df),
            'anomaly_days': sum(df['is_significant_anomaly'] if 'is_significant_anomaly' in df.columns else 0),
            'average_anomaly_score': df['combined_anomaly_score'].mean() if 'combined_anomaly_score' in df.columns else 0,
        }
        
        # Add summary for each indicator
        for indicator in self.fraud_indicators:
            if indicator in df.columns:
                summary[f'{indicator}_count'] = sum(df[indicator])
        
        # Get the most anomalous dates
        if 'combined_anomaly_score' in df.columns and 'date' in df.columns:
            top_anomalies = df.sort_values('combined_anomaly_score', ascending=False).head(5)
            for i, (_, row) in enumerate(top_anomalies.iterrows()):
                summary[f'top_anomaly_{i+1}_date'] = row['date'].strftime('%Y-%m-%d')
                summary[f'top_anomaly_{i+1}_score'] = row['combined_anomaly_score']
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame([summary])
        output_path = os.path.join(REPORTS_DIR, f"{symbol}_fraud_summary.csv")
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Saved fraud analysis summary to {output_path}")
        
        # 3. Generate a text report
        with open(os.path.join(REPORTS_DIR, f"{symbol}_fraud_report.txt"), 'w') as f:
            f.write(f"FRAUD ANALYSIS REPORT FOR {symbol}\n")
            f.write(f"Generated on: {summary['analysis_date']}\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total days analyzed: {summary['total_days_analyzed']}\n")
            f.write(f"Days with significant anomalies: {summary['anomaly_days']} ")
            f.write(f"({summary['anomaly_days']/summary['total_days_analyzed']*100:.1f}%)\n")
            f.write(f"Average anomaly score: {summary['average_anomaly_score']:.4f}\n\n")
            
            f.write("TOP ANOMALOUS DATES:\n")
            for i in range(1, 6):
                if f'top_anomaly_{i}_date' in summary:
                    f.write(f"{i}. {summary[f'top_anomaly_{i}_date']} - ")
                    f.write(f"Score: {summary[f'top_anomaly_{i}_score']:.4f}\n")
            
            f.write("\nFRAUD INDICATOR BREAKDOWN:\n")
            for indicator in self.fraud_indicators:
                if f'{indicator}_count' in summary:
                    f.write(f"{indicator}: {summary[f'{indicator}_count']} instances ")
                    f.write(f"({summary[f'{indicator}_count']/summary['total_days_analyzed']*100:.1f}%)\n")
            
            # Analysis conclusion based on thresholds
            f.write("\nANALYSIS CONCLUSION:\n")
            
            if summary['average_anomaly_score'] > 0.3:
                f.write("HIGH RISK: Significant anomalous activity detected\n")
                f.write("This stock shows multiple strong indicators of potential market manipulation or fraud.\n")
                f.write("Detailed investigation strongly recommended.\n")
            elif summary['average_anomaly_score'] > 0.1:
                f.write("MEDIUM RISK: Some anomalous activity detected\n")
                f.write("This stock shows some indicators of unusual market behavior that may warrant investigation.\n")
            else:
                f.write("LOW RISK: Minimal anomalous activity detected\n")
                f.write("This stock shows normal market behavior with few or no indicators of potential fraud.\n")


def main():
    """Main function to run fraud detection"""
    parser = argparse.ArgumentParser(description='Run fraud detection analysis for NEPSE stocks')
    parser.add_argument('--symbol', type=str, help='Stock symbol to analyze (default: all)')
    parser.add_argument('--report-only', action='store_true', help='Generate reports only without visualizations')
    
    args = parser.parse_args()
    
    analyzer = FraudAnalyzer()
    
    # Get list of available symbols
    if args.symbol:
        symbols = [args.symbol]
    else:
        # Get all symbols with processed data
        data_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_processed.csv"))
        symbols = [os.path.basename(file).split('_processed.csv')[0] for file in data_files]
        
        if not symbols:
            logger.error("No processed data found. Run preprocessing.py first.")
            return
    
    logger.info(f"Running fraud detection for symbols: {symbols}")
    
    for symbol in symbols:
        logger.info(f"Analyzing {symbol}")
        result_df = analyzer.analyze_stock(symbol)
        
        if result_df is not None:
            anomaly_count = sum(result_df['is_significant_anomaly']) if 'is_significant_anomaly' in result_df.columns else 0
            logger.info(f"Analysis complete for {symbol}. Found {anomaly_count} significant anomalies.")
    
    logger.info("Fraud detection completed")


if __name__ == "__main__":
    main()