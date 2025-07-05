import ccxt
import pandas as pd
import numpy as np
import seaborn as sns
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

# Get data and calculate results
print("Fetching data and calculating performance metrics...")
df = fetch_ohlcv('binance', 'BTC/USDT', '1h', '2021-01-01T00:00:00Z')

# Test a few parameter combinations to find the best one
test_params = [
    (200, 1.0),
    (400, 1.0),
    (600, 1.0),
    (800, 1.0),
    (1000, 1.0),
    (200, 1.5),
    (400, 1.5),
    (600, 1.5),
    (800, 1.5),
    (1000, 1.5)
]

best_sharpe = -np.inf
best_params = None
best_metrics = None

for window, threshold in test_params:
    print(f"Testing window={window}, threshold={threshold}")
    df_signal, trades = compute_signals(df.copy(), window, threshold)
    
    if len(trades) > 0:
        returns = calculate_strategy_returns(df_signal, trades)
        metrics = calculate_performance_metrics(returns)
        
        if metrics['Sharpe Ratio'] > best_sharpe:
            best_sharpe = metrics['Sharpe Ratio']
            best_params = (window, threshold)
            best_metrics = metrics

print(f"\n=== BEST PARAMETERS ===")
print(f"Window Size: {best_params[0]}")
print(f"Threshold: {best_params[1]}")
print(f"Sharpe Ratio: {best_sharpe:.4f}")

print(f"\n=== PERFORMANCE METRICS ===")
for metric, value in best_metrics.items():
    if isinstance(value, float):
        if 'Ratio' in metric:
            print(f"{metric}: {value:.4f}")
        elif 'Drawdown' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.2%}")
    else:
        print(f"{metric}: {value}")

# Calculate buy & hold performance for comparison
price_returns = df['close'].pct_change().fillna(0)
buy_hold_metrics = calculate_performance_metrics(price_returns)

print(f"\n=== BUY & HOLD COMPARISON ===")
print(f"Buy & Hold Total Return: {buy_hold_metrics['Total Return']:.2%}")
print(f"Buy & Hold Sharpe Ratio: {buy_hold_metrics['Sharpe Ratio']:.4f}")
print(f"Strategy vs Buy & Hold: {best_metrics['Total Return'] - buy_hold_metrics['Total Return']:.2%}") 