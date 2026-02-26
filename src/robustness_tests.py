import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core_metrics import find_rf_rate, find_annual_return
from pathlib import Path

def find_atr_cutoff(df, atr_cutoff):
	Window = 14
	df['TR'] = np.maximum(
		np.maximum(
			df['High'] - df['Low'],
			np.abs(df['High'] - df['Close'].shift(1))  
		),
		np.abs(df['Close'].shift(1) - df['Low'])
		)
	
	df['ATR'] = df['TR'].rolling(window=Window).mean()

	target_vol = df['ATR'].quantile(atr_cutoff)

	return target_vol

def walk_forward_analysis(prices_df, investment, short_ma = 20, long_ma = 50):
	windows = [
    {'name': 'Window 1', 'train': ('2015-01-01', '2018-12-31'), 'test': ('2019-01-01', '2020-12-31')},
    {'name': 'Window 2', 'train': ('2015-01-01', '2019-12-31'), 'test': ('2020-01-01', '2021-12-31')},
    {'name': 'Window 3', 'train': ('2015-01-01', '2020-12-31'), 'test': ('2021-01-01', '2022-12-31')},
    {'name': 'Window 4', 'train': ('2015-01-01', '2021-12-31'), 'test': ('2022-01-01', '2023-12-31')},
    {'name': 'Window 5', 'train': ('2015-01-01', '2022-12-31'), 'test': ('2023-01-01', '2024-12-31')},
    {'name': 'Window 6', 'train': ('2015-01-01', '2023-12-31'), 'test': ('2024-01-01', '2025-12-31')},
]

	results = []
	rolling_window = 14

	for window in windows:
		print(f"\n{window['name']}:") 
		print(f"  Train: {window['train'][0]} to {window['train'][1]}")
		print(f"  Test:  {window['test'][0]} to {window['test'][1]}")

		train_data = prices_df.loc[window['train'][0]:window['train'][1]].copy()
		test_data = prices_df.loc[window['test'][0]:window['test'][1]].copy()

		# calculate ATR thresholds for train periods
		train_data['TR'] = np.maximum(
			np.maximum(
				train_data['High'] - train_data['Low'],
				np.abs(train_data['High'] - train_data['Close'].shift(1))
			),
			np.abs(train_data['Close'].shift(1) - train_data['Low'])
		)
		train_data['ATR'] =  train_data['TR'].ewm(span=rolling_window, adjust=False).mean()
		train_data['ATR_pct'] = train_data['ATR'] / train_data['Close']

		ATR_threshold = train_data['ATR_pct'].quantile(0.7)
		print(f"  ATR threshold (from train): {ATR_threshold:.4f}")

		# run strategy on test data
		test_result = walk_forward_position_resize(investment, test_data, short_ma, long_ma, ATR_threshold)

		# calculate metrics
		# return
		final_val = test_result['portfolio_value'].iloc[-1]
		total_return = (final_val - investment) / investment 

		# risk free rate
		rf_return = find_rf_rate(test_data.index[0], test_data.index[-1])
		rf_daily = rf_return / 252

		# sharpe without costs
		daily_return_without_costs = test_result['strategy_ret'].dropna()
		sharpe_wo_costs = (daily_return_without_costs.mean() - rf_daily) / daily_return_without_costs.std()

		# sharpe with costs 
		daily_return_with_costs = test_result['strategy_ret_net'].dropna()
		sharpe_with_costs = (daily_return_with_costs.mean() - rf_daily) / daily_return_with_costs.std()

		# max drawdown
		peak = test_result['portfolio_value'].cummax()
		drawdown = (test_result['portfolio_value'] - peak) / peak
		max_dd = drawdown.min()

		# number of trades
		num_trades = (test_result['position'] != 0).sum()
		results.append({
            'window': window['name'],
            'train_start': train_data.index[0].date(),
            'train_end': train_data.index[-1].date(),
            'test_start': test_data.index[0].date(),
            'test_end': test_data.index[-1].date(),
            'atr_threshold': ATR_threshold,
            'total_return': total_return,
            'sharpe': sharpe_wo_costs,
			'sharpe_with_costs': sharpe_with_costs,
            'max_drawdown': max_dd,
            'num_trades': num_trades
        })

		print(f"\n  Test Results:")
		print(f"    Total Return: {total_return:.2%} (This is also annual return)")
		print(f"    Sharpe: {sharpe_wo_costs:.2f}")
		print(f"    Sharpe With Costs: {sharpe_with_costs:.2f}")
		print(f"    Max Drawdown: {max_dd:.2%}")
		print(f"    Trades: {num_trades}")
    
	results_df = pd.DataFrame(results)
	print(f"\n{'='*70}")
	print("WALK-FORWARD SUMMARY")
	print(f"{'='*70}")
	print(f"Average Sharpe Without Costs: {results_df['sharpe'].mean():.2f} (±{results_df['sharpe'].std():.2f})")
	print(f"Average Sharpe With Costs: {results_df['sharpe_with_costs'].mean():.2f} (±{results_df['sharpe_with_costs'].std():.2f})")
	print(f"Average Total/Annual Return: {results_df['total_return'].mean():.2%}")
	print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2%}")
	print(f"Sharpe Consistency: {results_df['sharpe'].std() / abs(results_df['sharpe'].mean()):.2f} (lower is better)")
	
	_plot_walk_forward_results(results_df)
	return results_df

def _plot_walk_forward_results(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Sharpe ratio across windows
    axes[0, 0].bar(results_df['window'], results_df['sharpe'], alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axhline(y=results_df['sharpe'].mean(), color='green', 
                        linestyle='--', linewidth=2, label=f"Avg: {results_df['sharpe'].mean():.2f}")
    axes[0, 0].set_title('Sharpe Ratio Across Windows')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Annual returns
    axes[0, 1].bar(results_df['window'], results_df['total_return'] * 100, 
                    alpha=0.7, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Annual/Total Return Across Windows')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Max drawdown
    axes[1, 0].bar(results_df['window'], results_df['max_drawdown'] * 100,
                    alpha=0.7, color='red')
    axes[1, 0].set_title('Max Drawdown Across Windows')
    axes[1, 0].set_ylabel('Max Drawdown (%)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Number of trades
    axes[1, 1].bar(results_df['window'], results_df['num_trades'],
                    alpha=0.7, color='purple')
    axes[1, 1].set_title('Number of Trades Across Windows')
    axes[1, 1].set_ylabel('Trade Count')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_folder = Path('../results/figures/')
    output_folder.mkdir(parents=True, exist_ok=True)  # Create folder if doesn't exist

    output_path = output_folder / 'walk_forward_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi=300 for high quality
    print(f'Figure saved to: {output_path}')
    plt.show()

    return

def walk_forward_position_resize(inv_amt, prices_df, short_ma, long_ma, 
                atr_cutoff,
                min_position=0.25,
                max_position=1.00,
                transaction_costs=0.001,
                slippage=0.0005):
    df = prices_df.copy()
    df = df.drop(columns=['Dividends', "Capital Gains", "Stock Splits"], errors="ignore") 

    df["MA_short"] = df["Close"].rolling(window=short_ma).mean()
    df["MA_long"] = df["Close"].rolling(window=long_ma).mean()

    window = 14

    # Calculate ATR (use EMA for consistency with V1)
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))  
        ),
        np.abs(df['Close'].shift(1) - df['Low'])
    )
    df['ATR'] = df['TR'].ewm(span=window, adjust=False).mean()  # FIXED: EMA
    df['ATR_pct'] = df['ATR'] / df['Close']

    # Target volatility
    target_vol = atr_cutoff

    # Position sizing
    df['position_size_ratio'] = np.clip(
        target_vol / df['ATR_pct'],
        min_position,
        max_position
    )

    # Generate signals
    df["signal"] = 0
    df.loc[(df["MA_short"] > df["MA_long"]) & (df['ATR'].notna()), "signal"] = 1

    # FIXED: Track actual position size (only set at entry, held until exit)
    df['actual_position_size'] = 0.0
    
    for i in range(len(df)):
        if i == 0:
            continue
            
        prev_signal = df.iloc[i-1]['signal']
        curr_signal = df.iloc[i]['signal']
        
        if curr_signal == 0:
            # Not in position
            df.iloc[i, df.columns.get_loc('actual_position_size')] = 0.0
        elif prev_signal == 0 and curr_signal == 1:
            # Entry - set position size based on current volatility
            df.iloc[i, df.columns.get_loc('actual_position_size')] = df.iloc[i]['position_size_ratio']
        else:
            # Holding - maintain same position size
            df.iloc[i, df.columns.get_loc('actual_position_size')] = df.iloc[i-1]['actual_position_size']

    # Calculate position changes
    df["position"] = df["signal"].diff()
    
    # Transaction costs
    total_cost = transaction_costs + slippage
    df["trade_cost"] = 0.0
    df.loc[df['position'] != 0, 'trade_cost'] = total_cost

    # Returns scaled by actual position size
    df['market_ret'] = df['Close'].pct_change()
    df['strategy_ret'] = df['market_ret'] * df['signal'].shift(1) * df['actual_position_size'].shift(1)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade_cost']
    
    # Market portfolio
    df['market_ret_net'] = df['market_ret'].copy()
    df.loc[df.index[0], 'market_ret_net'] = df.loc[df.index[0], 'market_ret'] - total_cost

    # Cumulative returns
    df['cumulative_market'] = (1 + df['market_ret_net']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_ret_net']).cumprod()
    
    # Portfolio value
    df['portfolio_value'] = inv_amt * df['cumulative_strategy']
    df['market_portfolio_value'] = inv_amt * df['cumulative_market']

    # Index handling
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index("Date", drop=True)
        else:
            df.index = pd.to_datetime(df.index)
   
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    return df

def parameter_sensitivity(strategy_fn, inv_amt, prices_df):
    """ 
    Test position sizing strategy performance across param variables 
    Test ±20-30% variation around chosen params 
    """
    ma_pairs = [
        (15, 40, "Short-term"),
        (18, 45, "Below baseline"),
        (20, 50, "Baseline"),  # Original chosen parameters
        (22, 55, "Above baseline"),
        (25, 60, "Long-term")
    ]
    
    atr_cutoffs = [0.60, 0.65, 0.70, 0.75, 0.80]  # Original baseline: 0.75
    
    results = []
    
    print("="*70)
    print("PARAMETER SENSITIVITY ANALYSIS - POSITION SIZING")
    print("="*70)

    # Test MA parameters
    print("\n1. Testing MA period sensitivity (atr_cutoff fixed at 0.75)...")
    for short, long, label in ma_pairs:
        result = strategy_fn(
            inv_amt=inv_amt,
            prices_df=prices_df,
            short_ma=short,
            long_ma=long,
            atr_cutoff=0.75
        )
        
        # Calculate metrics
        final_value = result['portfolio_value'].iloc[-1]
        total_return = (final_value - inv_amt) / inv_amt
        
        num_days = (result.index[-1] - result.index[0]).days
        num_years = num_days / 365.25
        annual_return = (final_value / inv_amt) ** (1/num_years) - 1
        
        rf_rate = find_rf_rate(result.index[0], result.index[-1])
        daily_returns = result['strategy_ret_net'].dropna()
        rf_daily = rf_rate / 252
        excess_returns = daily_returns - rf_daily
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        peak = result['portfolio_value'].cummax()
        drawdown = (result['portfolio_value'] - peak) / peak
        max_dd = drawdown.min()
        
        results.append({
            'parameter_type': 'MA Period',
            'short_ma': short,
            'long_ma': long,
            'atr_cutoff': 0.75,
            'label': label,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        })
        
        print(f"  MA({short},{long}): Sharpe={sharpe:.2f}, Return={annual_return:.2%}")

    # Test ATR cutoff threshold
    print("\n2. Testing ATR cutoff sensitivity (MA fixed at 20/50)...")
    for atr_cut in atr_cutoffs:
        result = strategy_fn(
            inv_amt=inv_amt,
            prices_df=prices_df,
            short_ma=20,
            long_ma=50,
            atr_cutoff=atr_cut
        )
        
        # Calculate metrics
        final_value = result['portfolio_value'].iloc[-1]
        total_return = (final_value - inv_amt) / inv_amt
        
        num_days = (result.index[-1] - result.index[0]).days
        num_years = num_days / 365.25
        annual_return = (final_value / inv_amt) ** (1/num_years) - 1
        
        rf_rate = find_rf_rate(result.index[0], result.index[-1])
        daily_returns = result['strategy_ret_net'].dropna()
        rf_daily = rf_rate / 252
        excess_returns = daily_returns - rf_daily
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        peak = result['portfolio_value'].cummax()
        drawdown = (result['portfolio_value'] - peak) / peak
        max_dd = drawdown.min()
        
        # Calculate average position size to show effect
        avg_position = result['actual_position_size'][result['signal'] == 1].mean()
        
        results.append({
            'parameter_type': 'ATR Cutoff',
            'short_ma': 20,
            'long_ma': 50,
            'atr_cutoff': atr_cut,
            'label': f"{atr_cut:.0%}",
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'avg_position_size': avg_position
        })
        
        print(f"  ATR Cutoff {atr_cut:.0%}: Sharpe={sharpe:.2f}, Return={annual_return:.2%}, Avg Position={avg_position:.1%}")

    results_df = pd.DataFrame(results)

    # Visualizations
    _plot_parameter_sensitivity_position_sizing(results_df)

    # Summary
    print("\n" + "="*70)
    print("SENSITIVITY SUMMARY")
    print("="*70)

    ma_results = results_df[results_df['parameter_type'] == 'MA Period']
    atr_results = results_df[results_df['parameter_type'] == 'ATR Cutoff']

    print(f"\nMA Period Sensitivity:")
    print(f"  Sharpe range: {ma_results['sharpe'].min():.2f} to {ma_results['sharpe'].max():.2f}")
    print(f"  Sharpe std dev: {ma_results['sharpe'].std():.2f}")
    print(f"  Conclusion: {'Robust' if ma_results['sharpe'].std() < 0.15 else 'Sensitive'}")

    print(f"\nATR Cutoff Sensitivity:")
    print(f"  Sharpe range: {atr_results['sharpe'].min():.2f} to {atr_results['sharpe'].max():.2f}")
    print(f"  Sharpe std dev: {atr_results['sharpe'].std():.2f}")
    if 'avg_position_size' in atr_results.columns:
        print(f"  Avg position range: {atr_results['avg_position_size'].min():.1%} to {atr_results['avg_position_size'].max():.1%}")
    print(f"  Conclusion: {'Robust' if atr_results['sharpe'].std() < 0.15 else 'Sensitive'}")

    return results_df


def _plot_parameter_sensitivity_position_sizing(results_df):
    """Visualize parameter sensitivity for position sizing strategy"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ma_results = results_df[results_df['parameter_type'] == 'MA Period']
    atr_results = results_df[results_df['parameter_type'] == 'ATR Cutoff']

    # MA Sharpe sensitivity
    axes[0, 0].plot(ma_results['short_ma'], ma_results['sharpe'], 
                    marker='o', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Sharpe vs MA Period')
    axes[0, 0].set_xlabel('Short MA Period')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].grid(True, alpha=0.3)

    # ATR cutoff Sharpe sensitivity
    axes[0, 1].plot(atr_results['atr_cutoff'] * 100, atr_results['sharpe'],
                    marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].axvline(x=75, color='blue', linestyle='--', linewidth=1, label='Baseline (75%)')
    axes[0, 1].set_title('Sharpe vs ATR Cutoff (Target Volatility)')
    axes[0, 1].set_xlabel('ATR Cutoff Percentile (%)')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MA Returns
    axes[1, 0].bar(ma_results['label'], ma_results['annual_return'] * 100, alpha=0.7)
    axes[1, 0].set_title('Annual Return vs MA Period')
    axes[1, 0].set_ylabel('Annual Return (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # ATR cutoff with dual axis (returns + avg position size)
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    # Bars for returns
    x = np.arange(len(atr_results))
    bars = ax1.bar(x, atr_results['annual_return'] * 100, 
                   alpha=0.7, color='green', label='Annual Return')
    ax1.set_ylabel('Annual Return (%)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_xticks(x)
    ax1.set_xticklabels(atr_results['label'])
    
    # Line for avg position size
    if 'avg_position_size' in atr_results.columns:
        line = ax2.plot(x, atr_results['avg_position_size'] * 100,
                       marker='o', linewidth=2, markersize=8, 
                       color='purple', label='Avg Position Size')
        ax2.set_ylabel('Avg Position Size (%)', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim([20, 105])  # Slightly above max position
    
    ax1.set_title('Return & Position Size vs ATR Cutoff')
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_folder = Path('../results/figures/')
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / 'parameter_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nFigure saved to: {output_path}')

    plt.show()


def regime_analysis(prices_df, strategy_func, inv_amt=10000, 
                   short_ma=20, long_ma=50, atr_cutoff=0.75):
    """
    Test strategy performance in different market regimes.
    
    Identifies when strategy works vs fails.
    """
    
    # Define regimes based on SPY history
    regimes = {
        'Bull Market 2016-2018': ('2016-01-01', '2018-12-31'),
        'Correction 2018': ('2018-10-01', '2018-12-31'),
        'Bull Market 2019': ('2019-01-01', '2019-12-31'),
        'COVID Crash 2020': ('2020-02-01', '2020-03-31'),
        'COVID Recovery': ('2020-04-01', '2020-12-31'),
        'Bull 2021': ('2021-01-01', '2021-12-31'),
        'Bear Market 2022': ('2022-01-01', '2022-12-31'),
        'Recovery 2023+': ('2023-01-01', '2024-12-31')
    }
    
    results = []
    
    print("="*70)
    print("MARKET REGIME ANALYSIS")
    print("="*70)
    
    for regime_name, (start, end) in regimes.items():
        try:
            regime_data = prices_df.loc[start:end]
            
            if len(regime_data) < 50:  # Not enough data
                continue
            
            print(f"\n{regime_name}:")
            print(f"  Period: {regime_data.index[0].date()} to {regime_data.index[-1].date()}")
            
            # Run strategy
            result = strategy_func(
                inv_amt=inv_amt,
                prices_df=regime_data,
                short_ma=short_ma,
                long_ma=long_ma,
                atr_cutoff=atr_cutoff
            )
            
            # Strategy metrics
            final_value = result['portfolio_value'].iloc[-1]
            num_days = (result.index[-1] - result.index[0]).days
            num_years = num_days / 365.25
            strategy_return = (final_value / inv_amt) ** (1/num_years) - 1
            
            # Buy & hold for comparison
            bnh_return = ((regime_data['Close'].iloc[-1] / regime_data['Close'].iloc[0]) 
                         ** (1/num_years) - 1)
            
            # Sharpe
            rf_rate = find_rf_rate(result.index[0], result.index[-1])
            daily_returns = result['strategy_ret_net'].dropna()
            rf_daily = rf_rate / 252
            excess_returns = daily_returns - rf_daily
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if len(excess_returns) > 0 else 0
            
            # Max drawdown
            peak = result['portfolio_value'].cummax()
            drawdown = (result['portfolio_value'] - peak) / peak
            max_dd = drawdown.min()
            
            # Outperformed?
            outperformed = strategy_return > bnh_return
            
            results.append({
                'regime': regime_name,
                'start': regime_data.index[0],
                'end': regime_data.index[-1],
                'strategy_return': strategy_return,
                'bnh_return': bnh_return,
                'outperformed': outperformed,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'num_days': num_days
            })
            
            print(f"  Strategy: {strategy_return:.2%} annual")
            print(f"  Buy&Hold: {bnh_return:.2%} annual")
            print(f"  {'✓ Outperformed' if outperformed else '✗ Underperformed'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Visualizations
    _plot_regime_analysis(results_df)
    
    # Summary
    print("\n" + "="*70)
    print("REGIME SUMMARY")
    print("="*70)
    print(f"Outperformed in {results_df['outperformed'].sum()} / {len(results_df)} regimes")
    print(f"Average outperformance: {(results_df['strategy_return'] - results_df['bnh_return']).mean():.2%}")
    
    return results_df


def _plot_regime_analysis(results_df):
    """Visualize regime analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Strategy vs Buy&Hold by regime
    x = np.arange(len(results_df))
    width = 0.35
    
    axes[0].bar(x - width/2, results_df['strategy_return'] * 100, 
                width, label='Strategy', alpha=0.8)
    axes[0].bar(x + width/2, results_df['bnh_return'] * 100,
                width, label='Buy & Hold', alpha=0.8)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df['regime'], rotation=45, ha='right')
    axes[0].set_title('Strategy vs Buy & Hold by Regime')
    axes[0].set_ylabel('Annual Return (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Sharpe by regime
    axes[1].bar(x, results_df['sharpe'], alpha=0.8, color='purple')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['regime'], rotation=45, ha='right')
    axes[1].set_title('Sharpe Ratio by Regime')
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # Save the figure
    output_folder = Path('../results/figures/')
    output_folder.mkdir(parents=True, exist_ok=True)  # Create folder if doesn't exist

    output_path = output_folder / 'regime_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi=300 for high quality
    print(f'Figure saved to: {output_path}')
    plt.show()


def monte_carlo_shuffle(result_df, inv_amt, n_simulations=1000):
    """
    Simple Monte Carlo: Shuffle return sequence.
    
    Tests: Was performance due to skill or lucky return ordering?
    
    Fast version - takes 5-10 seconds for 1000 simulations.
    """
    
    strategy_returns = result_df['strategy_ret_net'].dropna().values
    actual_final_value = result_df['portfolio_value'].iloc[-1]
    
    # Actual Sharpe
    rf_rate = find_rf_rate(result_df.index[0], result_df.index[-1])
    rf_daily = rf_rate / 252
    excess_returns = strategy_returns - rf_daily
    actual_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    print("="*70)
    print(f"MONTE CARLO SIMULATION ({n_simulations:,} runs)")
    print("="*70)
    print(f"Shuffling {len(strategy_returns)} daily returns...\n")
    
    simulated_values = []
    simulated_sharpes = []
    
    for i in range(n_simulations):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_simulations}")
        
        # Shuffle returns
        shuffled = np.random.permutation(strategy_returns)
        
        # Calculate final value
        cumulative = (1 + shuffled).cumprod()
        final_value = inv_amt * cumulative[-1]
        simulated_values.append(final_value)
        
        # Calculate Sharpe
        excess = shuffled - rf_daily
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
        simulated_sharpes.append(sharpe)
    
    simulated_values = np.array(simulated_values)
    simulated_sharpes = np.array(simulated_sharpes)
    
    # Calculate percentiles
    pct_beat_value = (simulated_values > actual_final_value).mean()
    pct_beat_sharpe = (simulated_sharpes > actual_sharpe).mean()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Final values
    axes[0].hist(simulated_values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=actual_final_value, color='red', linewidth=2,
                   label=f'Actual: ${actual_final_value:,.0f}')
    axes[0].axvline(x=inv_amt, color='black', linestyle='--', linewidth=2,
                   label=f'Initial: ${inv_amt:,.0f}')
    axes[0].set_title('Monte Carlo: Final Portfolio Values')
    axes[0].set_xlabel('Final Value ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sharpe ratios
    axes[1].hist(simulated_sharpes, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1].axvline(x=actual_sharpe, color='red', linewidth=2,
                   label=f'Actual: {actual_sharpe:.2f}')
    axes[1].set_title('Monte Carlo: Sharpe Ratios')
    axes[1].set_xlabel('Sharpe Ratio')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\nActual final value: ${actual_final_value:,.2f}")
    print(f"Simulated mean: ${simulated_values.mean():,.2f}")
    print(f"Simulated 5th percentile: ${np.percentile(simulated_values, 5):,.2f}")
    print(f"Simulated 95th percentile: ${np.percentile(simulated_values, 95):,.2f}")
    
    print(f"\nActual Sharpe: {actual_sharpe:.2f}")
    print(f"Simulated mean Sharpe: {simulated_sharpes.mean():.2f}")
    
    print(f"\n% of simulations beating actual value: {pct_beat_value:.1%}")
    print(f"% of simulations beating actual Sharpe: {pct_beat_sharpe:.1%}")
    
    if pct_beat_value < 0.05:
        print("\n✓ Result statistically significant (p < 0.05) - likely skill")
    else:
        print("\n✗ Result not significant - may be due to luck")
    
    return {
        'simulated_values': simulated_values,
        'simulated_sharpes': simulated_sharpes,
        'p_value_value': pct_beat_value,
        'p_value_sharpe': pct_beat_sharpe
    }

def monte_carlo_block_bootstrap(result_df, inv_amt, block_size=20, n_simulations=1000):
    """
    Block Bootstrap: Preserves autocorrelation in returns.
    
    Randomly samples BLOCKS of consecutive returns to maintain
    serial correlation (streaks of good/bad days).
    """
    
    strategy_returns = result_df['strategy_ret_net'].dropna().values
    n_days = len(strategy_returns)
    actual_final_value = result_df['portfolio_value'].iloc[-1]
    
    rf_rate = find_rf_rate(result_df.index[0], result_df.index[-1])
    rf_daily = rf_rate / 252
    excess_returns = strategy_returns - rf_daily
    actual_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    print("="*70)
    print(f"MONTE CARLO BLOCK BOOTSTRAP ({n_simulations:,} simulations)")
    print(f"Block size: {block_size} days")
    print("="*70)
    
    simulated_values = []
    simulated_sharpes = []
    
    # Create blocks
    n_blocks = n_days // block_size
    
    for i in range(n_simulations):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_simulations}")
        
        # Randomly sample blocks
        bootstrapped = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n_days - block_size)
            block = strategy_returns[start_idx:start_idx + block_size]
            bootstrapped.extend(block)
        
        bootstrapped = np.array(bootstrapped[:n_days])  # Trim to exact length
        
        # Calculate metrics
        cumulative = (1 + bootstrapped).cumprod()
        final_value = inv_amt * cumulative[-1]
        simulated_values.append(final_value)
        
        excess = bootstrapped - rf_daily
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
        simulated_sharpes.append(sharpe)
    
    simulated_values = np.array(simulated_values)
    simulated_sharpes = np.array(simulated_sharpes)
    
	# Calculate percentiles
    pct_beat_value = (simulated_values > actual_final_value).mean()
    pct_beat_sharpe = (simulated_sharpes > actual_sharpe).mean()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Final values
    axes[0].hist(simulated_values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=actual_final_value, color='red', linewidth=2,
                   label=f'Actual: ${actual_final_value:,.0f}')
    axes[0].axvline(x=inv_amt, color='black', linestyle='--', linewidth=2,
                   label=f'Initial: ${inv_amt:,.0f}')
    axes[0].set_title('Monte Carlo: Final Portfolio Values')
    axes[0].set_xlabel('Final Value ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sharpe ratios
    axes[1].hist(simulated_sharpes, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1].axvline(x=actual_sharpe, color='red', linewidth=2,
                   label=f'Actual: {actual_sharpe:.2f}')
    axes[1].set_title('Monte Carlo: Sharpe Ratios')
    axes[1].set_xlabel('Sharpe Ratio')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_folder = Path('../results/figures/')
    output_folder.mkdir(parents=True, exist_ok=True)  # Create folder if doesn't exist

    output_path = output_folder / 'monte_carlo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi=300 for high quality
    print(f'Figure saved to: {output_path}')
    plt.show()
    
    # Summary
    print(f"\nActual final value: ${actual_final_value:,.2f}")
    print(f"Simulated mean: ${simulated_values.mean():,.2f}")
    print(f"Simulated 5th percentile: ${np.percentile(simulated_values, 5):,.2f}")
    print(f"Simulated 95th percentile: ${np.percentile(simulated_values, 95):,.2f}")
    
    print(f"\nActual Sharpe: {actual_sharpe:.2f}")
    print(f"Simulated mean Sharpe: {simulated_sharpes.mean():.2f}")
    
    print(f"\n% of simulations beating actual value: {pct_beat_value:.1%}")
    print(f"% of simulations beating actual Sharpe: {pct_beat_sharpe:.1%}")
    
    if pct_beat_value < 0.05:
        print("\n✓ Result statistically significant (p < 0.05) - likely skill")
    else:
        print("\n✗ Result not significant - may be due to luck")
    
    return {
        'simulated_values': simulated_values,
        'simulated_sharpes': simulated_sharpes,
        'p_value_value': pct_beat_value,
        'p_value_sharpe': pct_beat_sharpe
    }