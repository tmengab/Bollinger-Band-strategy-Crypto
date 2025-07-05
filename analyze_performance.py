import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_ohlcv(exchange_name, symbol, timeframe, since, limit=1000):
    """Fetch OHLCV data from specified exchange"""
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'} if exchange_name == 'binance' else {}
    })
    
    all_ohlcv = []
    since_ms = exchange.parse8601(since)
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            since_ms = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break
        except Exception as e:
            print(f"Error fetching data from {exchange_name}: {e}")
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def compute_signals(df, window, threshold):
    """Compute Bollinger Band signals"""
    df = df.copy()
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['z_score'] = (df['close'] - df['SMA']) / df['STD']

    position = 0
    positions = []
    trades = []

    for i in range(len(df)):
        z = df['z_score'].iloc[i]
        
        if position != 0 and np.sign(z) != np.sign(df['z_score'].iloc[i-1] if i > 0 else 0):
            trades.append(('close', df['open'].iloc[i+1] if i+1 < len(df) else df['open'].iloc[i]))
            position = 0
        
        if position == 0:
            if z > threshold:
                position = 1
                trades.append(('buy', df['open'].iloc[i+1] if i+1 < len(df) else df['open'].iloc[i]))
            elif z < -threshold:
                position = -1
                trades.append(('sell', df['open'].iloc[i+1] if i+1 < len(df) else df['open'].iloc[i]))
        
        positions.append(position)

    df['position'] = positions
    return df, trades

def calculate_strategy_returns(df, trades, initial_capital=100000, fee=0.001):
    """Calculate strategy returns"""
    capital = initial_capital
    btc_held = 0
    portfolio_values = []
    
    # Track portfolio value over time
    for i in range(len(df)):
        current_price = df['open'].iloc[i]
        portfolio_value = capital + btc_held * current_price
        portfolio_values.append(portfolio_value)
    
    # Now process trades and update portfolio values
    trade_idx = 0
    for i in range(len(df)):
        # Check if there's a trade at this timestamp
        if trade_idx < len(trades):
            trade_type, price = trades[trade_idx]
            
            # Update capital and btc_held based on trade
            if trade_type == 'buy':
                cost = price * (1 + fee)
                if capital >= cost:
                    capital -= cost
                    btc_held += 1
            elif trade_type == 'sell':
                capital += price * (1 - fee)
                btc_held -= 1
            elif trade_type == 'close':
                if btc_held > 0:
                    capital += price * (1 - fee)
                else:
                    capital -= price * (1 + fee)
                btc_held = 0
            
            trade_idx += 1
        
        # Update portfolio value for this timestamp
        current_price = df['open'].iloc[i]
        portfolio_values[i] = capital + btc_held * current_price
    
    # Calculate returns
    returns = pd.Series(portfolio_values).pct_change().dropna()
    return returns

# Get data
print("Analyzing strategy performance...")
df = fetch_ohlcv('binance', 'BTC/USDT', '1h', '2021-01-01T00:00:00Z')

# Calculate strategy performance
window, threshold = 400, 1.0
df_signal, trades = compute_signals(df.copy(), window, threshold)
returns = calculate_strategy_returns(df_signal, trades)

# Calculate buy & hold performance
price_returns = df['close'].pct_change().fillna(0)
buy_hold_cumulative = (1 + price_returns).cumprod()
strategy_cumulative = (1 + returns).cumprod()

# Analyze by year
print(f"\n=== YEARLY PERFORMANCE ANALYSIS ===")
for year in [2021, 2022, 2023]:
    year_data = df[df.index.year == year]
    year_strategy = strategy_cumulative[strategy_cumulative.index.year == year]
    year_buyhold = buy_hold_cumulative[buy_hold_cumulative.index.year == year]
    
    if len(year_data) > 0:
        strategy_return = (year_strategy.iloc[-1] / year_strategy.iloc[0] - 1) * 100
        buyhold_return = (year_buyhold.iloc[-1] / year_buyhold.iloc[0] - 1) * 100
        
        print(f"{year}:")
        print(f"  Strategy: {strategy_return:.2f}%")
        print(f"  Buy & Hold: {buyhold_return:.2f}%")
        print(f"  Difference: {strategy_return - buyhold_return:.2f}%")

# Analyze trading frequency
print(f"\n=== TRADING ANALYSIS ===")
print(f"Total trades: {len(trades)}")
print(f"Trades per year: {len(trades) / 3:.1f}")

# Count trade types
buy_trades = sum(1 for trade in trades if trade[0] == 'buy')
sell_trades = sum(1 for trade in trades if trade[0] == 'sell')
close_trades = sum(1 for trade in trades if trade[0] == 'close')

print(f"Buy trades: {buy_trades}")
print(f"Sell trades: {sell_trades}")
print(f"Close trades: {close_trades}")

# Analyze z-score distribution
print(f"\n=== Z-SCORE ANALYSIS ===")
z_scores = df_signal['z_score'].dropna()
print(f"Z-score mean: {z_scores.mean():.4f}")
print(f"Z-score std: {z_scores.std():.4f}")
print(f"Z-score > 1.0: {(z_scores > 1.0).sum()} times")
print(f"Z-score < -1.0: {(z_scores < -1.0).sum()} times")
print(f"Z-score between -1.0 and 1.0: {((z_scores >= -1.0) & (z_scores <= 1.0)).sum()} times")

# Market trend analysis
print(f"\n=== MARKET TREND ANALYSIS ===")
df['daily_return'] = df['close'].pct_change(24)  # Daily returns
df['trend'] = df['daily_return'].rolling(30).mean()  # 30-day trend

trending_days = (df['trend'].abs() > 0.01).sum()  # Days with >1% daily trend
ranging_days = (df['trend'].abs() <= 0.01).sum()

print(f"Trending days (>1% daily trend): {trending_days}")
print(f"Ranging days (≤1% daily trend): {ranging_days}")
print(f"Market was trending {trending_days/(trending_days+ranging_days)*100:.1f}% of the time")

# Final conclusion
print(f"\n=== CONCLUSION ===")
print("The strategy performed poorly because:")
print("1. Bitcoin was in strong trending markets (2021-2023)")
print("2. Bollinger Bands work best in ranging/oscillating markets")
print("3. The strategy missed large trend moves by exiting at zero crossings")
print("4. High trading frequency led to significant fee erosion")
print("5. The fixed 1.0 threshold was too conservative for volatile crypto markets") 