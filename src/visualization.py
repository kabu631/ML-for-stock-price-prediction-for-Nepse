#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module for NEPSE Stock Market Analysis

This module provides visualization tools for stock data, predictions, and
fraud detection results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import glob
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
PLOTS_DIR = os.path.join("models", "plots")

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_processed_data(symbol):
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


def plot_stock_price_history(symbol, save_fig=True):
    """
    Plot stock price history with volume
    
    Args:
        symbol (str): Stock symbol
        save_fig (bool): Whether to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    df = load_processed_data(symbol)
    if df is None or df.empty:
        return None
    
    logger.info(f"Plotting stock price history for {symbol}")
    
    # Create figure with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot price on primary axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.plot(df['date'], df['close'], color='tab:blue', label='Close Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Add moving averages if they exist
    for ma in ['sma_20', 'sma_50']:
        if ma in df.columns:
            ax1.plot(df['date'], df[ma], label=ma.upper())
    
    # Create secondary y-axis for volume
    if 'volume' in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume', color='tab:red')
        ax2.bar(df['date'], df['volume'], alpha=0.3, color='tab:red', label='Volume')
        ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Add title and legend
    plt.title(f'{symbol} - Stock Price History')
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    if 'volume' in df.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(lines1, labels1, loc='upper left')
    
    # Save figure
    if save_fig:
        output_path = os.path.join(PLOTS_DIR, f"{symbol}_price_history.png")
        plt.savefig(output_path)
        logger.info(f"Saved price history plot to {output_path}")
    
    return fig


def plot_technical_indicators(symbol, save_fig=True):
    """
    Plot technical indicators
    
    Args:
        symbol (str): Stock symbol
        save_fig (bool): Whether to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    df = load_processed_data(symbol)
    if df is None or df.empty:
        return None
    
    logger.info(f"Plotting technical indicators for {symbol}")
    
    # Create a 3x2 subplot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{symbol} - Technical Indicators', fontsize=16)
    
    # 1. Price with Bollinger Bands
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['close'], label='Close Price')
    if all(x in df.columns for x in ['bollinger_high', 'bollinger_low', 'bollinger_mavg']):
        ax1.plot(df['date'], df['bollinger_high'], 'r--', label='Bollinger High')
        ax1.plot(df['date'], df['bollinger_low'], 'g--', label='Bollinger Low')
        ax1.plot(df['date'], df['bollinger_mavg'], 'b--', label='Bollinger MA')
    ax1.set_title('Price with Bollinger Bands')
    ax1.legend()
    ax1.grid(True)
    
    # 2. MACD
    ax2 = axes[0, 1]
    if all(x in df.columns for x in ['macd', 'macd_signal']):
        ax2.plot(df['date'], df['macd'], label='MACD')
        ax2.plot(df['date'], df['macd_signal'], label='Signal Line')
        if 'macd_diff' in df.columns:
            # Bar chart for MACD histogram
            positive_diff = df['macd_diff'] > 0
            ax2.bar(df.loc[positive_diff, 'date'], df.loc[positive_diff, 'macd_diff'], color='g', alpha=0.5)
            ax2.bar(df.loc[~positive_diff, 'date'], df.loc[~positive_diff, 'macd_diff'], color='r', alpha=0.5)
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # 3. RSI
    ax3 = axes[1, 0]
    if 'rsi' in df.columns:
        ax3.plot(df['date'], df['rsi'])
        ax3.axhline(y=70, color='r', linestyle='--')
        ax3.axhline(y=30, color='g', linestyle='--')
    ax3.set_title('RSI')
    ax3.grid(True)
    
    # 4. Volume and EMA
    ax4 = axes[1, 1]
    if 'volume' in df.columns:
        ax4.bar(df['date'], df['volume'], color='b', alpha=0.5, label='Volume')
        if 'volume_ema' in df.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df['date'], df['volume_ema'], 'r', label='Volume EMA')
    ax4.set_title('Volume')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Daily Returns
    ax5 = axes[2, 0]
    if 'daily_return' in df.columns:
        ax5.plot(df['date'], df['daily_return'], 'g-')
    ax5.set_title('Daily Returns')
    ax5.grid(True)
    
    # 6. Volatility
    ax6 = axes[2, 1]
    if 'volatility' in df.columns:
        ax6.plot(df['date'], df['volatility'], 'r-')
    ax6.set_title('Volatility (20-day)')
    ax6.grid(True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    if save_fig:
        output_path = os.path.join(PLOTS_DIR, f"{symbol}_technical_indicators.png")
        plt.savefig(output_path)
        logger.info(f"Saved technical indicators plot to {output_path}")
    
    return fig


def plot_correlation_matrix(symbol, save_fig=True):
    """
    Plot correlation matrix for features
    
    Args:
        symbol (str): Stock symbol
        save_fig (bool): Whether to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    df = load_processed_data(symbol)
    if df is None or df.empty:
        return None
    
    logger.info(f"Plotting correlation matrix for {symbol}")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Select most important features to avoid too cluttered plot
    important_features = ['open', 'high', 'low', 'close', 'volume', 
                         'sma_20', 'ema_12', 'macd', 'rsi', 'volatility', 
                         'daily_return', 'bollinger_high', 'bollinger_low']
    
    # Filter for columns that exist
    features = [col for col in important_features if col in numeric_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df[features].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title(f'{symbol} - Feature Correlation Matrix')
    
    # Save figure
    if save_fig:
        output_path = os.path.join(PLOTS_DIR, f"{symbol}_correlation_matrix.png")
        plt.savefig(output_path)
        logger.info(f"Saved correlation matrix plot to {output_path}")
    
    plt.tight_layout()
    return plt.gcf()


def plot_interactive_candlestick(symbol, days=90):
    """
    Create interactive candlestick chart using Plotly
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days to show
        
    Returns:
        plotly.graph_objects.Figure: Interactive figure
    """
    df = load_processed_data(symbol)
    if df is None or df.empty:
        return None
    
    logger.info(f"Creating interactive candlestick chart for {symbol}")
    
    # Get the last n days of data
    df = df.sort_values('date')
    df = df.tail(days)
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.02, row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    for ma, color in [('sma_20', 'blue'), ('sma_50', 'red')]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[ma],
                    name=ma.upper(),
                    line=dict(color=color)
                ),
                row=1, col=1
            )
    
    # Add volume
    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Stock Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        yaxis2_title='Volume',
        height=800,
    )
    
    # Save as HTML
    output_path = os.path.join(PLOTS_DIR, f"{symbol}_interactive_chart.html")
    fig.write_html(output_path)
    logger.info(f"Saved interactive chart to {output_path}")
    
    return fig


def main():
    """Main function for visualization"""
    parser = argparse.ArgumentParser(description='Visualize NEPSE stock data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to visualize')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--price', action='store_true', help='Generate price history visualization')
    parser.add_argument('--indicators', action='store_true', help='Generate technical indicators visualization')
    parser.add_argument('--correlation', action='store_true', help='Generate correlation matrix')
    parser.add_argument('--interactive', action='store_true', help='Generate interactive candlestick chart')
    
    args = parser.parse_args()
    
    if not args.all and not args.price and not args.indicators and not args.correlation and not args.interactive:
        # If no specific plot is requested, generate all
        args.all = True
    
    if args.all or args.price:
        plot_stock_price_history(args.symbol)
    
    if args.all or args.indicators:
        plot_technical_indicators(args.symbol)
    
    if args.all or args.correlation:
        plot_correlation_matrix(args.symbol)
    
    if args.all or args.interactive:
        plot_interactive_candlestick(args.symbol)
    
    logger.info(f"Visualizations for {args.symbol} completed")


if __name__ == "__main__":
    main()