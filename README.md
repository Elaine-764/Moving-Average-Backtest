# Moving Average Crossover Strategy: A Systematic Evaluation

**Research Question:** Can a simple moving average crossover strategy be made profitable through systematic refinement, or is the underlying concept fundamentally flawed?

**Answer:** No, not on highly efficient assets like SPY. Despite systematic refinements, the strategy consistently underperformed buy-and-hold across multiple market regimes.

---

## Key Findings

- **Baseline Strategy (MA 20/50):** 7.4542% annual return vs 11.0467% buy-and-hold (Sharpe: 0.41)
- **Three Refinements Tested:** ATR volatility filter, ADX regime detection, dynamic position sizing
- **Result:** All refinements failed to improve risk-adjusted returns
- **Root Cause:** Signal quality, not execution details - MA crossovers lack predictive power on liquid ETFs
- **Silver Lining:** Strategy provided downside protection during corrections (2018 Q4, 2022 bear market)

### Performance Summary

| Strategy           | Annual Return | Sharpe w/ Costs | Max Drawdown | Trades | vs Baseline     |
| ------------------ | ------------- | --------------- | ------------ | ------ | --------------- |
| Baseline           | 7.45%         | 0.48            | -21.30%      | 36     | —               |
| V1: ATR Filter     | 5.69%         | 0.37            | -17.52%      | 33     | Worse           |
| V2: ADX Regime     | 6.09%         | 0.38            | -21.30%      | 30     | Worse           |
| V3: Dynamic Sizing | 6.71%         | 0.44            | -20.91%      | 36     | Best Refinement |
| **Buy & Hold SPY** | **10.8%**     | **0.68**        | **-33.8%**   | **0**  | **Best**        |

## Read the Full Analysis

!!! **[DETAILED REPORT](results/DETAILED_REPORT.md)** - Complete methodology, findings, and robustness testing

The report includes:
- Comprehensive methodology and data description
- Detailed performance metrics for all strategies
- Walk-forward analysis (6 expanding windows)
- Parameter sensitivity testing
- Market regime analysis (7 different market conditions)
- Monte Carlo simulation (1000 bootstrap runs)
- Discussion of why refinements failed and what would be required for success
---

## Repository Structure
```
trading-strategy-backtest/
│
├── README.md                           
├── requirements.txt                    
├── .gitignore
│
├── data/
│   ├── raw/                           # Downloaded price data (SPY, Treasury)
│   │   ├── spy.csv
│   │   └── treasury_10yr.csv
│   ├── processed/                     
│   │   ├── spy_cleaned.csv
│   │   ├── training_data_70.csv       # 2015-2021 (training)
│   │   └── testing_data_30.csv        # 2022-2025 (validation)
│   └── strategy_results/              # Output from each strategy
│       ├── ma_baseline_strategy.csv
│       ├── ma_volatility_filter.csv
│       ├── ma_trend_detection.csv
│       └── ma_position_resizing.csv 
│
├── notebooks/                        # Analysis (run in order)
│   ├── 01_data_exploration.ipynb     # EDA, data quality checks
│   ├── 02_baseline_strategy.ipynb    # Basic MA(20,50) crossover
│   ├── 03_volatility_filter.ipynb    # Add ATR volatility filter
│   ├── 04_trend_detection.ipynb      # Add ADX regime detection
│   ├── 05_position_resize.ipynb      # Add dynamic position sizing
│   ├── 06_robustness_testing.ipynb   # Walk-forward, Monte Carlo, regime analysis
│   └──  07_graph_generation.ipynb    # generate graph
│
├── src/                              # Reusable Python modules
│   ├── strategies.py                 # All strategy implementations
│   ├── core_metrics.py               # Performance metrics (Sharpe, drawdown, etc)
│   ├── file_exports                  # File exporting functions
│   └── robustness_tests.py           # Walk-forward, parameter sensitivity, Monte Carlo
│
├── results/                           # Outputs and visualizations
│   ├── figures/                       # All plots and charts
│   │   ├── strategy_comparison.png
│   │   ├── parameter_sensitivity.png
│   │   ├── regime_analysis.png
│   │   ├── monte_carlo.
│   │   ├── out_of_sample.pngpng
│   │   └── walk_forward_results.png
└── └── DETAILED_REPORT.md            # **FULL ANALYSIS & FINDINGS**
```

---

## How to Reproduce

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/trading-strategy-backtest.git
cd trading-strategy-backtest
pip install -r requirements.txt
```

### 2. Run Notebooks in Order
```bash
jupyter notebook
```

Then execute notebooks 01-06 sequentially:
1. `01_data_exploration.ipynb` - Download and clean SPY data
2. `02_baseline_strategy.ipynb` - Test baseline MA(20,50)
3. `03_volatility_filter_refinement.ipynb ` - Add ATR volatility filter
4. `04_trend_detection.ipynb` - Add ADX regime detection  
5. `05_position_resize.ipynb` - Add dynamic position sizing
6. `06_robustness_testing.ipynb` - Full robustness analysis
7. (optional) `07_graph_generation.ipynb `

### 3. View Results

All figures saved to `results/figures/`  
Read full analysis in `results/DETAILED_REPORT.md`

---

## Technologies Used

- **Python 3.10+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing  
- **matplotlib - Visualization
- **yfinance** - Financial data

---

## Methodology Highlights

### Baseline Strategy
- **Entry:** Buy when 20-day MA crosses above 50-day MA
- **Exit:** Sell when 20-day MA crosses below 50-day MA
- **Position:** Fixed $10,000 per trade
- **Costs:** 0.15% per trade (commission + slippage)

### Refinements Tested
1. **V1: ATR Volatility Filter** - Avoid trading during top 30% most volatile periods
2. **V2: ADX Regime Detection** - Only trade when market is trending (ADX > 25)
3. **V3: Dynamic Position Sizing** - Scale position size inversely with volatility

### Robustness Testing
- **Train/Test Split:** 70/30 (2015-2021 train, 2022-2025 test)
- **Walk-Forward Analysis:** 6 expanding windows
- **Parameter Sensitivity:** Test MA periods ±20%, thresholds ±10%
- **Regime Analysis:** Test across 7 different market conditions
- **Monte Carlo:** 1000 bootstrap simulations

---

## Key Insights

### Why Refinements Failed

1. **Too Few Signals:** With only 25-36 trades over 7 years, filters either removed good trades or had insufficient statistical power
2. **Market Efficiency:** SPY is the most liquid, widely-followed asset globally - simple technical rules are arbitraged away
3. **Signal Quality Problem:** The issue isn't execution (costs, timing, sizing) but that MA crossovers lack predictive power
4. **Bull Market Dominance:** 2015-2021 was primarily bullish - strategies that keep you "out" hurt returns

### When It Works

Strategy shows value only during sharp corrections:
- **2018 Q4 Correction:** Avoided -45% drop
- **2022 Bear Market:** Lost -18% vs -19% for buy-and-hold

But chronic underperformance in bull markets (-5% annually) doesn't justify rare downside protection.

### What Would Be Required

For MA strategies to succeed:
1. **Less efficient assets** - Small-caps, emerging markets, crypto
2. **Multiple signals** - Ensemble approach combining fundamentals + technicals
3. **Machine learning** - Adaptive regime detection
4. **Different strategy** - Mean reversion may work where trend-following fails

---

## Value of Negative Results

Despite the strategy not beating buy-and-hold, this project demonstrates:

- Rigorous backtesting methodology with proper train/test splits  
- Understanding of overfitting risks  
- Comprehensive robustness testing across multiple dimensions  
- Transparency about strategy limitations  
- Clear documentation of the process

---

** If you found this analysis useful, please star this repo!**

---

*Last Updated: February 2025*
