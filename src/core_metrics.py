import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf

def find_num_years(df):

    start = pd.to_datetime(df.index[0])
    end = pd.to_datetime(df.index[-1])
    num_days = (end-start).days

    num_years = num_days/365.25
    return num_years

def find_annual_return(df, col):
    '''annual return'''
    num_years = find_num_years(df)

    cum_return = (1 + df[col]).prod() - 1
    annual_return = (1 + cum_return) ** (1/num_years) - 1
    return annual_return

def find_annual_volatility(df, col):
    '''annual volatility'''
    daily_volatility = df[col].std()
    annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
    
    return annualized_volatility

def pnl(df, investment_amount, strategy_name, benchmark_name, graph_title):
    # Raw, total returns
    final_portfolio_value = df['portfolio_value'].iloc[-1]
    print(f"Final portfolio value: {final_portfolio_value} on {df.index[-1]}")

    strategy_total_return = final_portfolio_value / investment_amount - 1
    print(f"Strategy returned {100*strategy_total_return}% from {df.index[0]} to {df.index[-1]}")

    market_total_return = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1
    print(f'Market returned {100 * market_total_return} from {df.index[0]} to {df.index[-1]}')

    # Annualized returns
    num_days = (df.index[-1] - df.index[0]).days
    num_years = num_days/365.25
    print(f'\nAnnualized across {num_years:.4f} years: ')

    # Strategy annualized return
    strategy_annual_ret = np.power(1 + strategy_total_return, 1/num_years) - 1
    print(f'{strategy_name} returned {(100 * strategy_annual_ret):.4f}% per year')


    # buy and hold market return
    market_annual_ret = np.power(1 + market_total_return, 1/num_years) - 1
    print(f'{benchmark_name} returned {(100 * market_annual_ret):.4f}% per year')

    # plotting
    plt.figure(figsize=(14, 7))

    df['market_portfolio_value'] = investment_amount*df['cumulative_market']
    plt.plot(df.index, df['portfolio_value'], label=strategy_name, linewidth=2)
    plt.plot(df.index, df['market_portfolio_value'], label=benchmark_name, linewidth=2)

    # Add initial investment line
    plt.axhline(y=investment_amount, color='black', linestyle='--', 
            linewidth=1, alpha=0.5, label='Initial Investment')

    plt.title(graph_title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('P&L ($1000s)', fontsize=12)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.1f}'))

    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df, strategy_annual_ret, market_annual_ret

def find_rf_rate(start_date, end_date):
    # fetch 10 yr bond data
    data = pd.read_csv("../data/raw/treasury_10yr.csv", index_col=0, parse_dates=True)
    partial_data = data.loc[start_date:end_date]
    risk_free_rate = partial_data['Close'].mean() / 100
    return risk_free_rate

def find_sp_500_ret(start_date, end_date):
    # fetch S&P 500 data from folder
    data = pd.read_csv("../data/raw/sp_500.csv", index_col=0, parse_dates=True)
    data = data.loc[start_date:end_date]
    sp_rate = data['Close'].mean() / 100
    num_years = (end_date - start_date).days / 252
    sp_rate_annualized = sp_rate / np.sqrt(num_years)
    return sp_rate_annualized

def downside_deviation(df, col, market_col):
    df['excess_ret'] = df[col] - df[market_col]
    df['excess_ret_neg'] = np.minimum(0, df['excess_ret']) ** 2

    downside_mean = df['excess_ret_neg'].mean()
    return np.sqrt(downside_mean)

def performance_ratios(df):
    # annual return without transaction costs
    return_without_costs = find_annual_return(df, 'strategy_ret')
    strategy_vol_without_costs = find_annual_volatility(df, 'strategy_ret')

    # annual return with transaction costs
    return_with_costs = find_annual_return(df, 'strategy_ret_net')
    strategy_vol_with_costs = find_annual_volatility(df, 'strategy_ret_net')

    ### WITHOUT TRANSACTION COSTS
    # risk free rate
    start_date = df.index[0]
    end_date = df.index[-1]
    rf_rate = find_rf_rate(start_date, end_date)

    # find strategy portfolio volatility
    # sharpe ratio
    sharpe_wo_cost = (return_without_costs - rf_rate)/strategy_vol_without_costs

    # Information ratio
    sp_500_return = find_sp_500_ret(start_date, end_date)
    information_ratio_wo_cost = (return_without_costs - sp_500_return)/ (strategy_vol_without_costs * (return_without_costs - sp_500_return))

    # Sortino ratio
    downside_dev = downside_deviation(df, 'strategy_ret', 'market_ret')
    sortino_wo_cost = (return_without_costs - rf_rate) / downside_dev

    ### WITH TRANSACTION COSTS

    # sharpe ratio
    sharpe_with_cost = (return_with_costs - rf_rate)/strategy_vol_with_costs

    # Information ratio
    sp_500_return = find_sp_500_ret(start_date, end_date)
    information_ratio_with_cost = (return_with_costs - sp_500_return)/ (strategy_vol_with_costs * (return_with_costs - sp_500_return))

    # Sortino ratio
    downside_dev = downside_deviation(df, 'strategy_ret_net', 'market_ret_net')
    sortino_with_cost = (return_with_costs - rf_rate) / downside_dev

    return {
        'sharpe ratio': sharpe_wo_cost,
        'information ratio': information_ratio_wo_cost,
        'sortino ratio': sortino_wo_cost,
        'sharpe with costs': sharpe_with_cost,
        'information ratio with costs': information_ratio_with_cost,
        'sortino ratio with costs': sortino_with_cost
    }

def risk_metrics(df):
    # drawdowns

    df['peak'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['peak'])/df['peak']
    max_drawdown = df['drawdown'].min()
    avg_drawdown = df['drawdown'].mean()

    df['in_dd'] = (df['drawdown'] < 0)
    df['drawdown_id'] = (df['in_dd'] != df['in_dd'].shift()).cumsum()

    drawdown_periods = df[df['in_dd']].groupby('drawdown_id').agg({
        'drawdown': 'min',  # Deepest point of this drawdown
        'portfolio_value': 'count'  # Number of days in drawdown
    }).rename(columns={'portfolio_value': 'duration'})

    max_drawdown_duration = drawdown_periods['duration'].max()
    avg_drawdown_duration = drawdown_periods['duration'].mean()

    volatility = find_annual_volatility(df, 'strategy_ret_net')

    return {
        'max_drawdown': max_drawdown, 
        'avg_drawdown': avg_drawdown,
        'max_dd_duration': max_drawdown_duration, 
        'avg_dd_duration': avg_drawdown_duration, 
        'volatility': volatility
    }

def trade_statistics(df):
    # Number of trades
    buy_signals = (df['position'] == 1).sum()
    sell_signal = (df['position'] == -1).sum()

    num_trades = buy_signals + sell_signal

    # accumulate a series of trades, with details
    trades = []
    entry_price = None
    entry_date = None

    for i, row in df.iterrows():
        if row['position'] == 1: # buy signal
            entry_price = row['Close']
            entry_date = i
        elif row['position'] == -1 and entry_price is not None:
            exit_price = row['Close']
            exit_date = i
            holding_days = (exit_date - entry_date).days
            pnl = (exit_price - entry_price) / entry_price # as a %
            trades.append({'pnl': pnl, 'days': holding_days})
            entry_price = None

    if len(trades) == 0:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_loss_ratio': average_win / averages_loss if averages_loss > 0 else np.inf,
            'profit_factor': 0,
            'avg_holding_period': 0
        }

	# Average holding period
    average_holding_period = np.mean([t['days'] for t in trades])
    
    # win rate
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] < 0]
    win_rate = len(wins) / (len(wins) + len(losses))
    
    # average win vs average loss
    average_win = np.mean(wins) if len(wins) > 0 else 0
    averages_loss = np.mean(losses) if len(losses) > 0 else 0

    # Profit factor (gross profit / gross loss)
    gross_profit = sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(sum(losses)) if len(losses) > 0 else 0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': average_win,
        'avg_loss': averages_loss,
        'win_loss_ratio': average_win / averages_loss if averages_loss > 0 else np.inf,
        'profit_factor': profit_factor,
        'avg_holding_period': average_holding_period
    }

def find_beta(df, start_date, end_date, col="strategy_ret_net",):
    # market
    market = pd.read_csv("../data/raw/spy.csv", index_col=0, parse_dates=True)

    market['Return'] = market['Close'].pct_change()
    aligned_market = market['Return'].reindex(df.index)
    market_var = aligned_market.var()
    beta = aligned_market.cov(df[col]) / market_var
    return beta

def capm(rf_rate, mkt_return, beta):
    capm = rf_rate + beta * (mkt_return - rf_rate)
    return capm
	
def find_alpha(df, annual_actual_return, annual_market_return):
    start_date = df.index[0]
    end_date = df.index[-1]
    beta = find_beta(df, start_date, end_date)
    rf = find_rf_rate(start_date, end_date)
    expected_return = capm(rf, annual_market_return, beta)
    return annual_actual_return - expected_return

def calmar_ratio(annualized_return, max_drawdown):
    return annualized_return / abs(max_drawdown)
    
def consistency(df, window = 252):
    # Monthly return distribution (histogram)
    df_monthly = df.resample('ME').agg({
        'portfolio_value': 'last',
        'market_portfolio_value': 'last'
    })

    df_monthly['strategy_monthly_ret'] = df_monthly['portfolio_value'].pct_change()
    df_monthly['market_monthly_ret'] = df_monthly['market_portfolio_value'].pct_change()

    # yearly returns
    df_yearly = df.resample('YE').agg({
        'portfolio_value': 'last',
        'market_portfolio_value': 'last'
    })
    
    df_yearly['strategy_yearly_ret'] = df_yearly['portfolio_value'].pct_change()
    df_yearly['market_yearly_ret'] = df_yearly['market_portfolio_value'].pct_change()

    # percentage of positive returns
    pct_positive_months = (df_monthly['strategy_monthly_ret'] > 0).sum() / len(df_monthly)
    pct_positive_years = (df_yearly['strategy_yearly_ret'] > 0).sum() / len(df_yearly)

	# - Rolling Sharpe ratio (is it consistent or just one good year?)
    df['rolling_returns'] = df['portfolio_value'].pct_change()
    df['rolling_sharpe'] = (
        df['rolling_returns'].rolling(window=window).mean() / 
        df['rolling_returns'].rolling(window=window).std()
    ) * np.sqrt(252)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Monthly return distribution
    axes[0, 0].hist(df_monthly['strategy_monthly_ret'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Monthly Return Distribution')
    axes[0, 0].set_xlabel('Monthly Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Yearly return distribution
    axes[0, 1].hist(df_yearly['strategy_yearly_ret'].dropna(), bins=15, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Yearly Return Distribution')
    axes[0, 1].set_xlabel('Yearly Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe ratio over time
    axes[1, 0].plot(df.index, df['rolling_sharpe'], linewidth=1.5)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_title('Rolling 1-Year Sharpe Ratio')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Comparison of monthly returns
    axes[1, 1].scatter(df_monthly['market_monthly_ret'], df_monthly['strategy_monthly_ret'], alpha=0.6)
    axes[1, 1].plot([-0.2, 0.2], [-0.2, 0.2], 'r--', linewidth=2, label='Equal performance')
    axes[1, 1].set_title('Strategy vs Market Monthly Returns')
    axes[1, 1].set_xlabel('Market Monthly Return')
    axes[1, 1].set_ylabel('Strategy Monthly Return')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return statistics
    return {
        'pct_positive_months': pct_positive_months,
        'pct_positive_years': pct_positive_years,
        'avg_monthly_return': df_monthly['strategy_monthly_ret'].mean(),
        'median_monthly_return': df_monthly['strategy_monthly_ret'].median(),
        'std_monthly_return': df_monthly['strategy_monthly_ret'].std(),
        'avg_yearly_return': df_yearly['strategy_yearly_ret'].mean(),
        'median_yearly_return': df_yearly['strategy_yearly_ret'].median(),
        'avg_rolling_sharpe': df['rolling_sharpe'].mean(),
        'min_rolling_sharpe': df['rolling_sharpe'].min(),
        'max_rolling_sharpe': df['rolling_sharpe'].max()
    }

def sensitivity(strategy_func, inv_amt, prices_df, short_ma, long_ma):
    # test strategy profitability at different transaction cost levels

    cost_level_bps = np.linspace(0, 50, 6)
    cost_levels = [c / 10000 for c in cost_level_bps]

    results = []

    for cost in cost_levels:
        result = strategy_func(inv_amt, prices_df, short_ma, long_ma, cost, slippage = 0.0005)
        final_value = result['portfolio_value'].iloc[-1]
        total_return = (final_value - inv_amt) / inv_amt

        num_years = find_num_years(result)
        annualized_return = (final_value / inv_amt) ** (1/num_years) -1

        daily_returns = result['strategy_ret_net']
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        results.append({
            'cost_bps': cost * 10000,
            'cost_pct': cost * 100,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'final_value': final_value
        })
    
    results_df = pd.DataFrame(results)
    # Find break-even point (where return becomes negative)
    break_even_idx = results_df[results_df['total_return'] < 0].index
    if len(break_even_idx) > 0:
        break_even_cost = results_df.loc[break_even_idx[0], 'cost_bps']
    else:
        break_even_cost = None

    def get_lims(series, padding=0.05):
        """Calculate y-axis limits with padding"""
        min_val = series.min()
        max_val = series.max()
        pad = (max_val - min_val) * padding
        return [min_val - pad, max_val + pad]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Total return vs cost
    axes[0, 0].plot(results_df['cost_bps'], results_df['total_return'] * 100, 
                    marker='o', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    if break_even_cost:
        axes[0, 0].axvline(x=break_even_cost, color='orange', linestyle='--', 
                          linewidth=2, label=f'Break-even: {break_even_cost:.0f} bps')
    axes[0, 0].set_title('Total Return vs Transaction Costs')
    axes[0, 0].set_xlabel('Transaction Cost (basis points)')
    axes[0, 0].set_ylabel('Total Return (%)')
    # axes[0, 0].set_ylim([40, 70]) 
    axes[0, 0].grid(True, alpha=0.3)
    # axes[0, 0].legend()
    
    # 2. Annualized return vs cost
    axes[0, 1].plot(results_df['cost_bps'], results_df['annualized_return'] * 100, 
                    marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Annualized Return vs Transaction Costs')
    axes[0, 1].set_xlabel('Transaction Cost (basis points)')
    axes[0, 1].set_ylabel('Annualized Return (%)')
    # axes[0, 1].set_ylim([4, 7]) 
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sharpe ratio vs cost
    axes[1, 0].plot(results_df['cost_bps'], results_df['sharpe_ratio'], 
                    marker='o', linewidth=2, markersize=8, color='purple')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Sharpe Ratio vs Transaction Costs')
    axes[1, 0].set_xlabel('Transaction Cost (basis points)')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    # axes[1, 0].set_ylim([.3, .65]) 
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Final portfolio value vs cost
    axes[1, 1].bar(results_df['cost_bps'], results_df['final_value'], 
                alpha=0.7, edgecolor='black')
    # axes[1, 1].axhline(y=inv_amt, color='red', linestyle='--', linewidth=2, label='Initial Investment')
    axes[1, 1].set_title('Final Portfolio Value vs Transaction Costs')
    axes[1, 1].set_xlabel('Transaction Cost (basis points)')
    axes[1, 1].set_ylabel('Final Value ($)')

    # Set y-axis range
    # axes[1, 1].set_ylim([13000, 18000]) 

    # axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 1. Total return vs cost
    axes[0, 0].set_ylim(get_lims(results_df['total_return'] * 100))

    # 2. Annualized return vs cost
    axes[0, 1].set_ylim(get_lims(results_df['annualized_return'] * 100))

    # 3. Sharpe ratio vs cost
    axes[1, 0].set_ylim(get_lims(results_df['sharpe_ratio']))

    # 4. Final portfolio value vs cost
    axes[1, 1].set_ylim(get_lims(results_df['final_value']))

    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Transaction Cost Sensitivity Analysis")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\n")
    if break_even_cost:
        print(f"Strategy breaks even at ~{break_even_cost:.0f} basis points")
    else:
        print("Strategy remains profitable at all tested cost levels")
    
    return results_df

# def all_core_metrics(df):
