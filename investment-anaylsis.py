"""
DESCRIPTION:
This script is my attempt at simulating an 'Investment Analysis Tool' that 
scores companies on various fundamentals, then does a naive backtest to see 
if a hypothetical portfolio outperforms the S&P 500.

I'm a high school student, so this is both a learning project and a demonstration
of how I think about data, code, and finance. Everything here is for educational 
purposes; it's not real investment advice!

FEATURES:
1) Mock data fetch for fundamentals (Revenue, SG&A, etc.).
2) Calculation of multiple ratios: SG&A efficiency, inventory turnover, net margin.
3) Creation of a "CompanyScore" that ranks firms.
4) A simple backtest that compares our top picks vs. the S&P 500.

DISCLAIMER:
- Financial data is randomly generated for demonstration. 
- The backtest logic is extremely simplified.
- Real investing is far more complex; I'm still in high school, but I'm excited 
  to learn about finance, data analysis, and coding.
"""

import pandas as pd
import numpy as np
import random
import sys
import datetime
from typing import Dict, Any

#  STEP 1: MOCKING OR FETCHING DATA
# ----------------------------------------------------------------------
# In a real scenario, I'd probably use yfinance or an API like
# Financial Modeling Prep to get actual data. But here, to keep it
# all offline, I'm generating random numbers that "look" like they 
# could be from real companies.

def generate_mock_financial_data(tickers: list) -> pd.DataFrame:
    """
    Generate random financial metrics for each ticker.
    
    FIELDS:
    - Revenue (somewhere between 1B and 200B)
    - SG&A (random fraction of Revenue)
    - COGS (random fraction of Revenue)
    - Inventory
    - NetIncome (calculated from Revenue - SG&A - COGS - random overhead)
    - MarketPrice (a random current stock price)
    
    This is obviously not real data. 
    I'm just simulating so we can test the code logic.
    """
    random_data = []
    for ticker in tickers:
        # Let's assume a random revenue from 1B to 200B
        revenue = random.uniform(1e9, 2e11)  
        
        # SG&A is typically 5%-30% of revenue, let's randomize that
        sga = revenue * random.uniform(0.05, 0.3)
        
        # COGS is often 20%-60% of revenue, let's randomize that
        cogs = revenue * random.uniform(0.2, 0.6)
        
        # Inventory is random. For big companies, it can be in the millions or billions
        inventory = random.uniform(1e7, 5e9)
        
        # Let's compute Net Income in a naive way
        # We'll subtract random "other overhead" to keep it uncertain
        overhead = random.uniform(1e6, 5e8)
        net_income = revenue - sga - cogs - overhead
        
        # MarketPrice is just a random guess for the stock's current price
        market_price = random.uniform(10, 500)
        
        random_data.append({
            'Ticker': ticker,
            'Revenue': revenue,
            'SGA': sga,
            'COGS': cogs,
            'Inventory': inventory,
            'NetIncome': net_income,
            'MarketPrice': market_price
        })
    
    return pd.DataFrame(random_data)

def generate_mock_historical_prices(tickers: list, 
                                    start_date: datetime.date, 
                                    end_date: datetime.date) -> Dict[str, pd.DataFrame]:
    """
    Generate random walk price data for each ticker in 'tickers' plus the S&P 500 index (^GSPC).
    
    This simulates daily closing prices from 'start_date' to 'end_date'.
    We'll store each DataFrame in a dictionary keyed by ticker symbol.
    
    This is not at all accurate for real markets. It's just so we
    can pretend to do a backtest. 
    """
    historical_data = {}
    
    # We'll also track the S&P 500 as '^GSPC'
    all_symbols = tickers + ['^GSPC']
    
    days_count = (end_date - start_date).days
    
    for symbol in all_symbols:
        # We'll start the price at a random level
        if symbol == '^GSPC':
            # Maybe let's start the S&P at ~4,000
            current_price = 4000.0
        else:
            current_price = random.uniform(10, 500)
        
        records = []
        for day_idx in range(days_count):
            # We treat each day as a random % move between -2% and +2%
            daily_change = random.uniform(-0.02, 0.02)
            current_price *= (1 + daily_change)
            
            record_date = start_date + datetime.timedelta(days=day_idx)
            records.append({
                'Date': record_date,
                'Close': current_price
            })
        
        df_prices = pd.DataFrame(records)
        df_prices['Ticker'] = symbol
        historical_data[symbol] = df_prices
    
    return historical_data

#  STEP 2: CALCULATE RATIOS AND SCORES
# ----------------------------------------------------------------------
# This is where I do my "secret sauce" to see which companies 
# might be more interesting to consider. It's super naive, but it 
# shows the concept: I'd combine multiple fundamental metrics 
# into one "score."

def calculate_ratios_and_score(financials: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns for:
    1) SG&A Efficiency: SGA / Revenue
    2) Inventory Turnover: COGS / Inventory
    3) Net Margin: NetIncome / Revenue
    4) (Optional) Price/Earnings: MarketPrice / (NetIncome / # shares)
       but we don't have # shares, so let's skip it for now. :)
    5) CompanyScore: Weighted mixture of the above
    """
    df = financials.copy()
    
    # We have to watch for possible negative or zero inventory or revenue, 
    # but let's skip that complexity for now
    df['SGA_Efficiency'] = df['SGA'] / df['Revenue']          # Lower is better
    df['InventoryTurnover'] = df['COGS'] / df['Inventory']    # Higher is better
    df['NetMargin'] = df['NetIncome'] / df['Revenue']         # Higher is better
    
    # Let's define a naive scoring formula:
    # Score = 0.3*(1 - SGA_Efficiency) + 0.3*(InventoryTurnover / (1+InvTurnover)) + 0.4*(NetMargin)
    # Rationale: 
    # - We invert SGA_Efficiency because smaller = better
    # - We "squash" InventoryTurnover using x/(1+x) so it doesn't blow up if it's huge
    # - We rely more heavily on NetMargin (0.4 weight)
    
    # NOTE: This formula is random, there's no proof it correlates with real returns.
    df['CompanyScore'] = (
        0.3 * (1 - df['SGA_Efficiency']) +
        0.3 * (df['InventoryTurnover'] / (1 + df['InventoryTurnover'])) +
        0.4 * (df['NetMargin'])
    )
    
    return df

# ----------------------------------------------------------------------
#  STEP 3: BUILD A PORTFOLIO
# ----------------------------------------------------------------------

def build_portfolio(df_scored: pd.DataFrame, top_n: int = 5) -> list:
    """
    Sort companies by 'CompanyScore' descending and pick the top N tickers.
    """
    sorted_df = df_scored.sort_values('CompanyScore', ascending=False)
    selected = sorted_df.head(top_n)['Ticker'].tolist()
    return selected

# ----------------------------------------------------------------------
#  STEP 4: RUN A SIMPLE BACKTEST
# ----------------------------------------------------------------------
# This is extremely simplified. We do an "equal-weight" portfolio
# across our selected tickers, then compare how it evolves 
# vs. the S&P 500 index over the same time period.

def naive_backtest(price_data: Dict[str, pd.DataFrame], portfolio_tickers: list) -> Dict[str, Any]:
    """
    Compare the portfolio's returns vs. the S&P 500.
    We assume:
      - We start with $X and split it equally among all portfolio tickers at the first day's price.
      - We track daily 'Close' for each symbol.
      - We track the S&P 500 as '^GSPC' in the same date range.
      - We measure final value vs. initial, turning it into a % return.
      
    Returns a dictionary with:
      - Start date, end date
      - Portfolio final return (%)
      - S&P 500 final return (%)
    """
    
    # We'll try to merge daily price data across all portfolio tickers 
    # plus the S&P 500. For each, we store 'Date' and 'Close'.
    
    # Let's pick one ticker's DataFrame as our reference for date range.
    if not portfolio_tickers:
        return {'error': 'No tickers selected for portfolio!'}
    
    # We'll choose the first ticker's date range to "base" our DataFrame
    base_ticker = portfolio_tickers[0]
    base_df = price_data[base_ticker].copy()
    base_df.rename(columns={'Close': f'Close_{base_ticker}'}, inplace=True)
    
    # Merge others
    merged_df = base_df[['Date', f'Close_{base_ticker}']]
    
    for t in portfolio_tickers[1:]:
        df_t = price_data[t][['Date', 'Close']].copy()
        df_t.rename(columns={'Close': f'Close_{t}'}, inplace=True)
        merged_df = pd.merge(merged_df, df_t, on='Date', how='inner')
    
    # Merge the S&P 500
    sp_df = price_data['^GSPC'][['Date', 'Close']].copy()
    sp_df.rename(columns={'Close': 'Close_SP500'}, inplace=True)
    merged_df = pd.merge(merged_df, sp_df, on='Date', how='inner')
    
    merged_df.sort_values('Date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # Calculate the portfolio's daily value
    # We do equal weighting. If we have N tickers, we put 1/N of the capital in each.
    ticker_count = len(portfolio_tickers)
    weight = 1.0 / ticker_count
    
    # The first day's portfolio value = sum of (Close_T * weight)
    # For simplicity, let's assume we buy exactly 1 share for each 'weight portion' 
    # but let's do it in $ terms to keep it consistent.
    
    initial_portfolio = 0.0
    for t in portfolio_tickers:
        initial_portfolio += merged_df.loc[0, f'Close_{t}'] * weight
    
    # The first day's S&P value
    initial_sp = merged_df.loc[0, 'Close_SP500']
    
    final_portfolio = 0.0
    final_sp = 0.0
    
    # We'll look at the last row to see final values
    last_idx = len(merged_df) - 1
    for t in portfolio_tickers:
        final_portfolio += merged_df.loc[last_idx, f'Close_{t}'] * weight
    
    final_sp = merged_df.loc[last_idx, 'Close_SP500']
    
    # Compute returns
    portfolio_return = (final_portfolio / initial_portfolio) - 1
    sp_return = (final_sp / initial_sp) - 1
    
    # We'll store some info
    result = {
        'start_date': merged_df.loc[0, 'Date'],
        'end_date': merged_df.loc[last_idx, 'Date'],
        'tickers_in_portfolio': portfolio_tickers,
        'portfolio_return_pct': round(portfolio_return*100, 2),
        'sp500_return_pct': round(sp_return*100, 2)
    }
    
    return result

# ----------------------------------------------------------------------
#  STEP 5: MAIN EXECUTION FLOW (for local testing)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Let's pick a random set of 10 tickers
    sample_tickers = ['AAPL', 'TSLA', 'WMT', 'MSFT', 'AMZN',
                      'GOOGL', 'NVDA', 'KO', 'JNJ', 'META']
    
    # 1) Generate mock fundamentals
    fundamentals_df = generate_mock_financial_data(sample_tickers)
    
    # 2) Compute ratios & scores
    scored_df = calculate_ratios_and_score(fundamentals_df)
    
    # Let's see how each company scored (for debugging)
    print("\n--- FUNDAMENTALS & SCORES ---")
    print(scored_df[['Ticker','Revenue','SGA_Efficiency','InventoryTurnover','NetMargin','CompanyScore']])
    
    # 3) Build a top-5 portfolio
    top_5 = build_portfolio(scored_df, top_n=5)
    print(f"\nTop 5 Tickers by 'CompanyScore': {top_5}\n")
    
    # 4) Generate mock historical price data from Jan 1 to July 1
    start_dt = datetime.date(2023, 1, 1)
    end_dt = datetime.date(2023, 7, 1)
    hist_prices = generate_mock_historical_prices(sample_tickers, start_dt, end_dt)
    
# 5) Run a naive backtest
backtest_result = naive_backtest(hist_prices, top_5)

print("\n--- BACKTEST RESULT ---")
if 'error' not in backtest_result:
    print(f"Start Date: {backtest_result['start_date']}")
    print(f"End Date: {backtest_result['end_date']}")
    print(f"Portfolio Tickers: {backtest_result['tickers_in_portfolio']}")
    print(f"Portfolio Return: {backtest_result['portfolio_return_pct']}%")
    print(f"S&P 500 Return: {backtest_result['sp500_return_pct']}%")
else:
    print("No portfolio built. Something went wrong!")

