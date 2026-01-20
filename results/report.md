


Research question: Can a simple moving average crossover strategy be made profitable through systematic refinement, or is the underlying concept fundamentally flawed?

  

## Methodology

### Data

- Assets tested: SPY (S&P 500 ETF)

- Period: 2015-2025 (11 years)

- Train/test split: 70/30 - backtest on the first 70% of data, paper test on the latter 30% of data

- Data source: Yahoo Finance via yfinance

  

### Baseline Strategy

- Entry: Buy when 50-day MA crosses above 200-day MA

- Exit: Sell when 50-day MA crosses below 200-day MA

- Position sizing: Fixed $10,000 per trade

- Transaction costs: 0.1% per trade (conservative estimate)

  

### Evaluation Metrics

Two "rounds" of evaluations:
#### I. Core performance metrics 
These are performed for the basic strategy and for every refinement.
- P&L Metrics
	- Total return (%)
	- Annualized return
	- Cumulative P&L curve (visual)
- Risk-Adjusted Performance
	- Sharpe Ratio 
	- Information Ratio - *comparing to a specific benchmark strategy*??
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
- Benchmark Comparison
	- Buy-and-hold SPY return over same period
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
- Out-of-Sample check: test strategy on the latter 30% of the data
#### II. Robustness testing
Done on the basic strategy and the best performing refinement. 
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

2. V2: Regime detection (trending vs ranging markets)

3. V3: Dynamic position sizing based on volatility

  

## Results

  

## Failure Cases

  

## How to Improve