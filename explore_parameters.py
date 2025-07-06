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

# Get data
print("Fetching data and exploring parameter combinations...")
df = fetch_ohlcv('binance', 'BTC/USDT', '1h', '2021-01-01T00:00:00Z')

# Current best parameters
current_best_window = 400
current_best_threshold = 1.0

# Explore nearby parameter combinations
print(f"Current best: Window={current_best_window}, Threshold={current_best_threshold}")
print("Exploring nearby combinations...")

# Window sizes around 400
window_sizes = [300, 350, 400, 450, 500, 550, 600]
# Thresholds around 1.0
thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

results = []

for window in window_sizes:
    for threshold in thresholds:
        print(f"Testing window={window}, threshold={threshold}")
        
        try:
            df_signal, trades = compute_signals(df.copy(), window, threshold)
            
            if len(trades) > 0:
                returns = calculate_strategy_returns(df_signal, trades)
                metrics = calculate_performance_metrics(returns)
                
                results.append({
                    'window': window,
                    'threshold': threshold,
                    'trades': len(trades),
                    'total_return': metrics['Total Return'],
                    'cagr': metrics['CAGR'],
                    'sharpe': metrics['Sharpe Ratio'],
                    'max_drawdown': metrics['Max Drawdown'],
                    'calmar': metrics['Calmar Ratio']
                })
                
                print(f"  Sharpe: {metrics['Sharpe Ratio']:.4f}, Return: {metrics['Total Return']:.2%}")
            else:
                print(f"  No trades generated")
                
        except Exception as e:
            print(f"  Error: {e}")

# Find different types of best combinations
if results:
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    print(f"\n=== PARAMETER ANALYSIS ===")
    print(f"Total combinations tested: {len(results_df)}")
    
    # 1. Best Sharpe Ratio (Risk-adjusted returns)
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    print(f"\n🥇 BEST SHARPE RATIO:")
    print(f"Window Size: {best_sharpe['window']}")
    print(f"Threshold: {best_sharpe['threshold']}")
    print(f"Sharpe Ratio: {best_sharpe['sharpe']:.4f}")
    print(f"Total Return: {best_sharpe['total_return']:.2%}")
    print(f"Max Drawdown: {best_sharpe['max_drawdown']:.2%}")
    print(f"Trades: {best_sharpe['trades']}")
    
    # 2. Best Total Return (Absolute returns)
    best_return = results_df.loc[results_df['total_return'].idxmax()]
    print(f"\n💰 BEST TOTAL RETURN:")
    print(f"Window Size: {best_return['window']}")
    print(f"Threshold: {best_return['threshold']}")
    print(f"Total Return: {best_return['total_return']:.2%}")
    print(f"Sharpe Ratio: {best_return['sharpe']:.4f}")
    print(f"Max Drawdown: {best_return['max_drawdown']:.2%}")
    print(f"Trades: {best_return['trades']}")
    
    # 3. Best Calmar Ratio (Return per unit of drawdown)
    best_calmar = results_df.loc[results_df['calmar'].idxmax()]
    print(f"\n📈 BEST CALMAR RATIO:")
    print(f"Window Size: {best_calmar['window']}")
    print(f"Threshold: {best_calmar['threshold']}")
    print(f"Calmar Ratio: {best_calmar['calmar']:.4f}")
    print(f"Total Return: {best_calmar['total_return']:.2%}")
    print(f"Max Drawdown: {best_calmar['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {best_calmar['sharpe']:.4f}")
    print(f"Trades: {best_calmar['trades']}")
    
    # 4. Lowest Max Drawdown (Conservative)
    best_drawdown = results_df.loc[results_df['max_drawdown'].idxmax()]  # Note: max_drawdown is negative, so idxmax gives least negative
    print(f"\n🛡️ LOWEST MAX DRAWDOWN:")
    print(f"Window Size: {best_drawdown['window']}")
    print(f"Threshold: {best_drawdown['threshold']}")
    print(f"Max Drawdown: {best_drawdown['max_drawdown']:.2%}")
    print(f"Total Return: {best_drawdown['total_return']:.2%}")
    print(f"Sharpe Ratio: {best_drawdown['sharpe']:.4f}")
    print(f"Trades: {best_drawdown['trades']}")
    
    # 5. Best Risk-Return Trade-off (Custom metric)
    # Define a custom score that balances return and risk
    results_df['risk_return_score'] = results_df['total_return'] / abs(results_df['max_drawdown'])
    best_risk_return = results_df.loc[results_df['risk_return_score'].idxmax()]
    print(f"\n⚖️ BEST RISK-RETURN TRADE-OFF:")
    print(f"Window Size: {best_risk_return['window']}")
    print(f"Threshold: {best_risk_return['threshold']}")
    print(f"Risk-Return Score: {best_risk_return['risk_return_score']:.4f}")
    print(f"Total Return: {best_risk_return['total_return']:.2%}")
    print(f"Max Drawdown: {best_risk_return['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {best_risk_return['sharpe']:.4f}")
    print(f"Trades: {best_risk_return['trades']}")
    
    # 6. Most Active Trading (High frequency)
    most_trades = results_df.loc[results_df['trades'].idxmax()]
    print(f"\n🔄 MOST ACTIVE TRADING:")
    print(f"Window Size: {most_trades['window']}")
    print(f"Threshold: {most_trades['threshold']}")
    print(f"Trades: {most_trades['trades']}")
    print(f"Total Return: {most_trades['total_return']:.2%}")
    print(f"Sharpe Ratio: {most_trades['sharpe']:.4f}")
    print(f"Max Drawdown: {most_trades['max_drawdown']:.2%}")
    
    # 7. Least Active Trading (Low frequency)
    least_trades = results_df.loc[results_df['trades'].idxmin()]
    print(f"\n⏰ LEAST ACTIVE TRADING:")
    print(f"Window Size: {least_trades['window']}")
    print(f"Threshold: {least_trades['threshold']}")
    print(f"Trades: {least_trades['trades']}")
    print(f"Total Return: {least_trades['total_return']:.2%}")
    print(f"Sharpe Ratio: {least_trades['sharpe']:.4f}")
    print(f"Max Drawdown: {least_trades['max_drawdown']:.2%}")
    
    # Summary table
    print(f"\n📊 SUMMARY COMPARISON:")
    summary_data = [
        ['Best Sharpe', best_sharpe['window'], best_sharpe['threshold'], best_sharpe['sharpe'], best_sharpe['total_return'], best_sharpe['max_drawdown'], best_sharpe['trades']],
        ['Best Return', best_return['window'], best_return['threshold'], best_return['sharpe'], best_return['total_return'], best_return['max_drawdown'], best_return['trades']],
        ['Best Calmar', best_calmar['window'], best_calmar['threshold'], best_calmar['sharpe'], best_calmar['total_return'], best_calmar['max_drawdown'], best_calmar['trades']],
        ['Lowest DD', best_drawdown['window'], best_drawdown['threshold'], best_drawdown['sharpe'], best_drawdown['total_return'], best_drawdown['max_drawdown'], best_drawdown['trades']],
        ['Best Risk-Return', best_risk_return['window'], best_risk_return['threshold'], best_risk_return['sharpe'], best_risk_return['total_return'], best_risk_return['max_drawdown'], best_risk_return['trades']],
        ['Most Trades', most_trades['window'], most_trades['threshold'], most_trades['sharpe'], most_trades['total_return'], most_trades['max_drawdown'], most_trades['trades']],
        ['Least Trades', least_trades['window'], least_trades['threshold'], least_trades['sharpe'], least_trades['total_return'], least_trades['max_drawdown'], least_trades['trades']]
    ]
    
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Window', 'Threshold', 'Sharpe', 'Return', 'MaxDD', 'Trades'])
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Recommendations based on different risk profiles
    print(f"\n💡 RECOMMENDATIONS BY RISK PROFILE:")
    print(f"🟢 Conservative (Low Risk): Window={best_drawdown['window']}, Threshold={best_drawdown['threshold']}")
    print(f"🟡 Balanced (Risk-Adjusted): Window={best_sharpe['window']}, Threshold={best_sharpe['threshold']}")
    print(f"🔴 Aggressive (High Return): Window={best_return['window']}, Threshold={best_return['threshold']}")
    print(f"⚖️ Optimal Trade-off: Window={best_risk_return['window']}, Threshold={best_risk_return['threshold']}")
    
    # Compare with current best
    current_best = next((r for r in results if r['window'] == current_best_window and r['threshold'] == current_best_threshold), None)
    
    if current_best:
        print(f"\n=== COMPARISON WITH CURRENT BEST ===")
        print(f"Current Best Sharpe: {current_best['sharpe']:.4f}")
        print(f"New Best Sharpe: {best_result['sharpe']:.4f}")
        print(f"Improvement: {best_result['sharpe'] - current_best['sharpe']:.4f}")
        
        if best_result['sharpe'] > current_best['sharpe']:
            print("🎉 Found better parameters!")
        else:
            print("Current parameters are still the best.")
    
    # Create results table
    print(f"\n=== ALL RESULTS (sorted by Sharpe ratio) ===")
    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    print(f"{'Window':<8} {'Threshold':<10} {'Sharpe':<8} {'Return':<10} {'Drawdown':<10} {'Trades':<8}")
    print("-" * 60)
    
    for i, result in enumerate(sorted_results[:10]):  # Top 10
        print(f"{result['window']:<8} {result['threshold']:<10.1f} {result['sharpe']:<8.4f} "
              f"{result['total_return']:<10.2%} {result['max_drawdown']:<10.2%} {result['trades']:<8}")
    
else:
    print("No valid results found.") 