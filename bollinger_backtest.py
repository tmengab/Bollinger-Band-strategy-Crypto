import ccxt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_ohlcv(exchange_name, symbol, timeframe, since, limit=1000, params=None):
    """Fetch OHLCV data from specified exchange"""
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'} if exchange_name == 'binance' else {}
    })
    
    if params:
        exchange.params = params
        
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

    position = 0  # 1 for long, -1 for short, 0 for flat
    positions = []
    trades = []

    for i in range(len(df)):
        z = df['z_score'].iloc[i]
        
        # Close position if z-score crosses zero
        if position != 0 and np.sign(z) != np.sign(df['z_score'].iloc[i-1] if i > 0 else 0):
            trades.append(('close', df['open'].iloc[i+1] if i+1 < len(df) else df['open'].iloc[i]))
            position = 0
        
        # Open new position if flat and threshold crossed
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

def calculate_strategy_returns(df, trades, initial_capital=100000, fee=0.001):  # 0.1% fee as required
    """Calculate strategy returns based on trades"""
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

def calculate_sharpe_ratio(returns, annualization_factor=24*365):
    """Calculate annualized Sharpe ratio (assuming risk-free rate = 0)"""
    return np.sqrt(annualization_factor) * returns.mean() / returns.std()

def generate_sharpe_heatmap(df, x_vals, y_vals):
    """Generate Sharpe ratio heatmap for parameter combinations"""
    sharpe_matrix = np.zeros((len(x_vals), len(y_vals)))
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            df_signal, trades = compute_signals(df.copy(), x, y)
            if len(trades) > 0:  # Only calculate if there were trades
                returns = calculate_strategy_returns(df_signal, trades)
                sharpe = calculate_sharpe_ratio(returns)
                sharpe_matrix[i, j] = sharpe
            else:
                sharpe_matrix[i, j] = np.nan
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(sharpe_matrix, 
                xticklabels=[f"{y:.2f}" for y in y_vals], 
                yticklabels=x_vals,
                cmap='viridis')
    plt.xlabel('Z-score threshold (y)')
    plt.ylabel('Window size (x)')
    plt.title('Sharpe Ratio Heatmap for Bollinger Band Strategy')
    plt.tight_layout()
    plt.savefig('sharpe_heatmap.png')
    plt.show()
    
    return sharpe_matrix

def calculate_performance_metrics(returns, capital=100000):
    """Calculate comprehensive performance metrics"""
    # Total Return
    total_return = (returns + 1).prod() - 1

    # CAGR
    days = len(returns) / 24  # since hourly data, convert to days
    years = days / 365
    cagr = (1 + total_return) ** (1 / years) - 1

    # Sharpe Ratio (assume risk-free rate = 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Calmar Ratio = CAGR / abs(Max Drawdown)
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar
    }

def main():
    # Parameters
    symbol = 'BTC/USDT'
    timeframe = '1h'
    start_date = '2021-01-01T00:00:00Z'  # Full period as required: Jan 1, 2021 to Dec 31, 2023
    
    # Fetch data from both exchanges
    print("Fetching data from Binance...")
    binance_data = fetch_ohlcv('binance', symbol, timeframe, start_date)
    
    print("Fetching data from Coinbase...")
    coinbase_data = fetch_ohlcv('coinbase', 'BTC-USD', timeframe, start_date)
    
    # For this example, we'll use Binance data for the strategy
    # In a real implementation, you might want to compare or combine both datasets
    df = binance_data
    
    # FAST VERSION: Reduced parameter ranges for quick testing
    x_vals = range(200, 1201, 200)  # Window sizes: 200, 400, 600, 800, 1000
    y_vals = np.arange(0.5, 2.51, 0.5)  # Thresholds: 0.5, 1.0, 1.5, 2.0, 2.5
    
    # Generate heatmap
    print("Generating Sharpe ratio heatmap...")
    sharpe_matrix = generate_sharpe_heatmap(df, x_vals, y_vals)
    
    # Find best parameters
    best_idx = np.unravel_index(np.nanargmax(sharpe_matrix), sharpe_matrix.shape)
    best_window = x_vals[best_idx[0]]
    best_threshold = y_vals[best_idx[1]]
    
    print(f"\nBest parameters - Window: {best_window}, Threshold: {best_threshold:.2f}")
    
    # Calculate performance with best parameters
    df_signal, trades = compute_signals(df.copy(), best_window, best_threshold)
    returns = calculate_strategy_returns(df_signal, trades)
    metrics = calculate_performance_metrics(returns)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'Ratio' in metric:
                print(f"{metric}: {value:.2f}")
            elif 'Drawdown' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()