


Research question: Can a simple moving average crossover strategy be made profitable through systematic refinement, or is the underlying concept fundamentally flawed?

  

## Methodology

### Data

- Assets tested: SPY (S&P 500 ETF)

- Period: 2015-2025 (11 years)

- Train/test split: 70/30 - backtest on the first 70% of data, paper test on the latter 30% of data

- Data source: Yahoo Finance via yfinance

  

### Baseline Strategy

- Entry: Buy when 20-day MA crosses above 30-day MA
	- reasoning for 20/50 split - tested 50-200 before, too few number of trades. 20/50 gives statistically meaningful number of trades and reasonal holding periods 

- Exit: Sell when 50-day MA crosses below 200-day MA

- Position sizing: Fixed $10,000 per trade

- Transaction costs: 0.1% per trade (conservative estimate)

- Price slippage while transacting: 0.05% per trade due to bid-ask spreads and market impact of trade
  

### Evaluation Metrics

Two "rounds" of evaluations:
#### I. Core performance metrics 
These are performed for the basic strategy and for every refinement.
- P&L Metrics
	- Total return (%)
	- Annualized return
	- Cumulative P&L curve (visual)
- Performance Ratio
	- Sharpe Ratio - Compare with risk free return rate, calculated as the return rate of the 10 year US bond annualized over the time frame.
	- Information Ratio - compare against S&P 500 returns
	- Sortino Ratio - bonus, penalizes downside volatility only
- Risk Metrics
	- Maximum Drawdown (%)
	- Average Drawdown
	- Drawdown duration (how long to recover)
	- Volatility (annualized std dev) 
- Trade Statistics
	- Win Rate (%)
	- Average win vs average loss
	- Profit factor (gross profit / gross loss)
	- Number of trades
	- Average holding period
- Strategy beta to benchmark
	- Alpha (excess return over benchmark)
	- Calmar Ratio (return / max drawdown)
- Return Consistency
	- Monthly/quarterly return distribution (histogram)
	- Rolling Sharpe ratio (is it consistent or just one good year?)
	- Percentage of positive months/years
- Transaction Cost Sensitivity
	- Test at multiple cost levels: 0 bps, 5 bps, 10 bps, 20 bps
	- Show where strategy breaks even
	- Include slippage estimates (bid-ask spread)
#### II. Robustness testing
Done on the best performing refinement. 
- Out-of-Sample check: test strategy on the latter 30% of the data
- Walk-Forward Analysis
	- train on year 1-3, test on year 4-5
	- train again on year 6-9, test on years 10-11
- Parameter sensitivity
- Different market regimes
	- Test separately on: Bull markets (2016-2019), Bear markets (2022), High volatility (2020), Low volatility (2017)
- Monte Carlo Simulation
### Refinement Approach

Iteratively added filters and tested on training set:

1. V1: Volatility filter (ATR-based)
	- threshold is don't trade in the top 30% most volatile times
	- ATR (Average True Range) rolling window = 14 days

2. V2: Regime detection (trending vs ranging markets)
	- only trade when market is trending, as measured by Welles Wilder's Average Directional Movement Index (ADX) - trade when ADX > 20
	- EMA preferred because it responds to market changes more rapidly, less lag, it's a standard in Wilder's formula

3. V3: Dynamic position sizing based on volatility: varying the position size based on volatility
	- position size dependent on the following formula (inspired by the Kelly Criterion): 
	Position size = Base_capital × (Target_volatility / Current_volatility)
		- Target_volatility is the comfort level, in this case the 75th percentile of the ATR in the training period. This only reduces position in times of extreme volatility
	- minimum position is 30% of total capital, maximum is 100%
	- volatility depends on ATR, trade more when it's less risky

  

## Results


## Limitations
- taxes not considered
- trades are executed at closing prices, which may not be most realistic. Might be more realistic to execute at next day's open
- full liquidity (may not be applicable for other assets)
- partial shares allowed
  

## Failure Cases

  

## How to Improve