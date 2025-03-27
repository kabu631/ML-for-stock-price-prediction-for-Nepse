# NEPSE Stock Analysis Notebooks

This directory contains Jupyter notebooks for analyzing NEPSE stock data using the AI-powered Finance and Fraud Detection system.

## Available Notebooks

Create the following notebooks in this directory:

1. **NEPSE_Stock_Analysis.ipynb**: Main demo notebook that showcases:
   - Data collection from NEPSE
   - Data preprocessing and feature engineering
   - LSTM model training for stock price prediction
   - Making future price predictions
   - Fraud detection using an autoencoder model
   - Visualization of stock data and analysis results

2. **Technical_Analysis.ipynb**: Detailed technical analysis of NEPSE stocks, including:
   - Moving averages (SMA, EMA)
   - MACD (Moving Average Convergence Divergence)
   - RSI (Relative Strength Index)
   - Bollinger Bands
   - Volume analysis

3. **Fraud_Detection_Analysis.ipynb**: In-depth analysis of potential fraud patterns:
   - Anomaly detection with autoencoders
   - Statistical anomaly detection
   - Price manipulation pattern detection
   - Visualization of suspicious activities

## Usage

To use these notebooks:

1. Make sure all dependencies are installed:
   ```
   pip install -r ../requirements.txt
   ```

2. Run the data collection script first:
   ```
   python ../src/data_collection.py
   ```

3. Open the notebooks with Jupyter:
   ```
   jupyter notebook
   ```

4. Follow the steps in each notebook to analyze NEPSE stock data.

## Creating Your Own Analysis

You can use these notebooks as templates for your own analysis:

1. Copy one of the existing notebooks and rename it
2. Modify the parameters and stock symbols
3. Run the analysis for your specific needs

Happy analyzing! 