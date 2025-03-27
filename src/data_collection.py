#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEPSE Stock Data Collection Module

This module provides functionality for collecting historical stock data from the
Nepal Stock Exchange (NEPSE) market using web scraping and API calls.
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
import json
import logging
from tqdm import tqdm
import urllib3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/nepse_data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = os.path.join("data", "raw")
NEPSE_URL = "https://www.nepalstock.com/"
NEPSE_API_URL = "https://www.nepalstock.com/api/nots/market-data"
COMPANY_LIST_URL = "https://www.nepalstock.com/api/nots/company"


class NepseDataCollector:
    """Class to collect and manage NEPSE stock data"""
    
    def __init__(self):
        """Initialize the NEPSE data collector"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification for demo purposes
        # Suppress SSL warnings for demo
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Ensure data directory exists
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    def get_company_list(self):
        """Fetch the list of companies listed on NEPSE"""
        try:
            logger.info("Fetching company list from NEPSE")
            
            # For demo: Generate sample company data instead of API call
            logger.info("Using simulated company data for demo")
            sample_companies = [
                {"symbol": "ADBL", "name": "Agricultural Development Bank Limited"},
                {"symbol": "NABIL", "name": "Nabil Bank Limited"},
                {"symbol": "NICA", "name": "NIC Asia Bank Ltd."},
                {"symbol": "NRIC", "name": "Nepal Reinsurance Company Limited"},
                {"symbol": "NTC", "name": "Nepal Telecom"},
                {"symbol": "NEPSE", "name": "Nepal Stock Exchange"},
                {"symbol": "NICL", "name": "Nepal Insurance Co. Ltd."},
                {"symbol": "PRVU", "name": "Prabhu Bank Limited"},
                {"symbol": "PCBL", "name": "Prime Commercial Bank Ltd."},
                {"symbol": "SCB", "name": "Standard Chartered Bank Nepal Ltd."}
            ]
            df = pd.DataFrame(sample_companies)
            
            # Save to CSV
            output_path = os.path.join(RAW_DATA_DIR, "nepse_companies.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} companies to {output_path}")
            return df
            
            # Original API code commented out for now
            # response = self.session.get(COMPANY_LIST_URL, headers=self.headers)
            # response.raise_for_status()
            # data = response.json()
            
            # if data.get('status') == 'success':
            #     companies = data.get('data', [])
            #     df = pd.DataFrame(companies)
            #     
            #     # Save to CSV
            #     output_path = os.path.join(RAW_DATA_DIR, "nepse_companies.csv")
            #     df.to_csv(output_path, index=False)
            #     logger.info(f"Saved {len(df)} companies to {output_path}")
            #     return df
            # else:
            #     logger.error(f"API returned error: {data.get('message')}")
            #     return None
                
        except Exception as e:
            logger.error(f"Error fetching company list: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, start_date=None, end_date=None):
        """
        Fetch historical data for a specific company
        
        Args:
            symbol (str): Company symbol/code
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        
        try:
            # This is a placeholder for actual API call - NEPSE doesn't provide simple historical data API
            # In a real implementation, you'd have to use either web scraping or their specific API endpoints
            
            # For demonstration, we'll create a simulated data structure
            # In a real implementation, replace this with actual API calls or web scraping
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            data = []
            
            for date in dates:
                # Simulate price data - in a real implementation, this would come from an API/web
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'open': 100 + (hash(f"{symbol}_{date.strftime('%Y-%m-%d')}_open") % 100),
                    'high': 110 + (hash(f"{symbol}_{date.strftime('%Y-%m-%d')}_high") % 100),
                    'low': 90 + (hash(f"{symbol}_{date.strftime('%Y-%m-%d')}_low") % 100),
                    'close': 105 + (hash(f"{symbol}_{date.strftime('%Y-%m-%d')}_close") % 100),
                    'volume': 1000 + (hash(f"{symbol}_{date.strftime('%Y-%m-%d')}_volume") % 10000),
                })
            
            df = pd.DataFrame(data)
            output_path = os.path.join(RAW_DATA_DIR, f"{symbol}_historical.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved historical data for {symbol} to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def get_market_summary(self):
        """Get the latest market summary from NEPSE"""
        try:
            logger.info("Fetching market summary")
            
            # For demo: Generate sample market data instead of API call
            logger.info("Using simulated market data for demo")
            market_data = {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "index": 2158.49,
                "change": 12.34,
                "percent_change": 0.58,
                "turnover": 1345678900,
                "transaction_count": 16789,
                "traded_shares": 3456789,
                "market_cap": 345678901234,
            }
            
            # Save to JSON
            output_path = os.path.join(RAW_DATA_DIR, "market_summary.json")
            with open(output_path, 'w') as f:
                json.dump(market_data, f, indent=4)
            logger.info(f"Saved market summary to {output_path}")
            return market_data
            
            # Original API code commented out for now
            # response = self.session.get(NEPSE_API_URL, headers=self.headers)
            # response.raise_for_status()
            # data = response.json()
            # 
            # if data.get('status') == 'success':
            #     market_data = data.get('data', {})
            #     # Save to JSON
            #     output_path = os.path.join(RAW_DATA_DIR, "market_summary.json")
            #     with open(output_path, 'w') as f:
            #         json.dump(market_data, f, indent=4)
            #     logger.info(f"Saved market summary to {output_path}")
            #     return market_data
            # else:
            #     logger.error(f"API returned error: {data.get('message')}")
            #     return None
                
        except Exception as e:
            logger.error(f"Error fetching market summary: {str(e)}")
            return None
    
    def fetch_all_company_data(self, days=365):
        """
        Fetch historical data for all companies
        
        Args:
            days (int): Number of days of historical data to fetch
            
        Returns:
            dict: Dictionary of company symbols to dataframes
        """
        companies = self.get_company_list()
        if companies is None or len(companies) == 0:
            logger.error("No companies found to fetch data for")
            return {}
            
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        results = {}
        for index, company in tqdm(companies.iterrows(), total=len(companies), desc="Fetching company data"):
            symbol = company.get('symbol')
            if not symbol:
                continue
                
            logger.info(f"Processing {symbol} ({index+1}/{len(companies)})")
            df = self.get_historical_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                results[symbol] = df
            
            # Be nice to the server - add delay between requests
            time.sleep(1)
            
        return results


def main():
    """Main function to run the data collection process"""
    logger.info("Starting NEPSE data collection")
    collector = NepseDataCollector()
    
    # Get company list
    companies = collector.get_company_list()
    if companies is not None:
        logger.info(f"Found {len(companies)} companies")
    
    # Get market summary
    market_summary = collector.get_market_summary()
    if market_summary is not None:
        logger.info("Market summary collected successfully")
    
    # Fetch data for top 10 companies (to avoid too many requests for demo)
    if companies is not None and len(companies) > 0:
        top_companies = companies.head(10)
        for index, company in top_companies.iterrows():
            symbol = company.get('symbol')
            if symbol:
                collector.get_historical_data(symbol)
    
    logger.info("Data collection completed")


if __name__ == "__main__":
    main() 